"""Callable prompt extension entities backed by HeavenBase workspaces."""

from __future__ import annotations

__all__ = ["Prompt", "ensure_prompt_entities"]

import inspect
from time import time

from heavenbase.capsule import Capsule, CapsuleManifest
from heavenbase.entity import field
from heavenbase.types import Float, Json, LongText, ShortText
from heavenbase.utils import Any, Callable, Dict, List, Optional, sha256hash

from ._workspace import prompt_config, resolve_prompt_workspace
from .name import check_prompt_name, parse_prompt_ref, prompt_row_id
from .translation import Translation


def _template_code(name: str, template: str, tr_keys: List[str]) -> str:
    return f'''def render(*args, tr=str, **kwargs):
    """Render a string prompt template."""
    values = dict(kwargs)
    if args:
        if len(args) == 1 and hasattr(args[0], "items"):
            values = dict(args[0]) | values
        else:
            raise TypeError("Template prompts accept keyword args or one mapping positional arg")
    template = tr({template!r})
    for key in {list(tr_keys)!r}:
        if key in values:
            values[key] = tr(str(values[key]))
    try:
        return str(template).format(**values)
    except KeyError as error:
        raise KeyError(f"Prompt {name!r} missing template placeholder {{error.args[0]!r}}") from error
    except Exception as error:
        raise ValueError(f"Prompt {name!r} template render failed: {{error}}") from error
'''


class Prompt(Capsule):
    """Callable prompt object and ``sys-prompt`` entity.

    ``Prompt`` wraps a Python callable or format-string template, exposes it as a normal callable object, and can persist
    the callable metadata as a HeavenBase entity row. Persisted prompts restore through Capsule manifests so function
    prompts and string templates use the same runtime path.

    Warnings:
        Persisted prompts contain executable Capsule source. Only load prompt rows from trusted workspaces.

    Examples:
        >>> prompt = Prompt("Hello, {name}", name="demo.hello")
        >>> prompt(name="Ada")
        'Hello, Ada'
    """

    identifier = "sys-prompt"

    name = field(ShortText).desc("Dotted prompt name")
    version = field(ShortText).default("1").desc("Explicit prompt version")
    status = field(ShortText).default("active").desc("Prompt status")
    description = field(LongText).default("").desc("Prompt description")
    tags = field(Json).default([]).desc("Prompt tags")
    manifest = field(Json).default({}).desc("Prompt manifest")
    capsule = field(Json).default({}).desc("Capsule manifest")
    dicts = field(Json).default([]).desc("Ordered translation dict names")
    main_lang = field(ShortText).default("en").desc("Fallback language")
    fingerprints = field(Json).default({}).desc("Integrity fingerprints")
    created_at = field(Float).default(0.0).desc("Creation timestamp")
    updated_at = field(Float).default(0.0).desc("Last update timestamp")

    def __init__(
        self,
        source: Optional[Any] = None,
        *,
        name: Optional[str] = None,
        version: str = "1",
        register: bool = False,
        ws: Any = None,
        dicts: Optional[List[str]] = None,
        tr_keys: Optional[List[str]] = None,
        main_lang: Optional[str] = None,
        lang: Optional[str] = None,
        elicit: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        **meta: Any,
    ) -> None:
        """Create a function-like prompt.

        Args:
            source (Any | None): Python callable, format-string template, or ``None`` for decorator/load-time binding.
            name (str | None): Dotted prompt name. Defaults to the callable name, or ``"prompt"`` for string templates.
            version (str): Explicit prompt version. Defaults to ``"1"``.
            register (bool): Whether to persist immediately after binding. Defaults to ``False``.
            ws (Any | None): Workspace used for registration, loading, and translations. Defaults to the active workspace.
            dicts (List[str] | None): Ordered translation dictionaries. Defaults to the prompt name.
            tr_keys (List[str] | None): Template argument names whose values should be translated before formatting.
            main_lang (str | None): Source language. Defaults to ``heavenbase.prompt.main_lang``.
            lang (str | None): Runtime language override for calls made through this prompt object.
            elicit (str | None): Missing-translation policy. ``"none"`` returns source text; ``"llm"`` is reserved.
            description (str): Prompt description. Defaults to the callable docstring when available.
            tags (List[str] | None): Prompt tags stored in the manifest.
            **meta (Any): Extra JSON-compatible metadata stored in the prompt manifest.

        Warnings:
            Loading or calling a persisted prompt restores and executes Capsule source code. Treat prompt workspaces as
            code registries and only load prompts from trusted stores.

        Examples:
            >>> prompt = Prompt("Hello, {name}", name="demo.hello")
            >>> prompt(name="Ada")
            'Hello, Ada'

        Returns:
            None: This initializer does not return a value.
        """
        inferred_name = getattr(source, "__name__", None) if callable(source) else None
        prompt_name = check_prompt_name(name or inferred_name or "prompt") if name or source is not None else None
        prompt_version = str(version or "1")
        prompt_description = description or str(getattr(source, "__doc__", "") or "").strip()
        prompt_tags = list(tags or [])
        config = prompt_config(ws)
        super().__init__(
            None,
            name=prompt_name,
            namespace="prompt",
            version=prompt_version,
            description=prompt_description,
            tags=prompt_tags,
            layers=("source",),
            resolver=getattr(ws, "module_resolver", None),
            config=config,
        )
        self.name = prompt_name
        self.version = prompt_version
        self.description = prompt_description
        self.tags = prompt_tags
        self.meta = dict(meta)
        self.dicts = list(dicts or ([self.name] if self.name else []))
        self._main_lang_defaulted = not bool(main_lang)
        self._elicit_defaulted = not bool(elicit)
        self.main_lang = main_lang or str(config.get("heavenbase.prompt.main_lang", default="en") or "en")
        self.lang = lang
        self.elicit = elicit or str(config.get("heavenbase.prompt.elicit", default="none") or "none")
        self.ws = ws
        self.tr = Translation(self)
        self._register_on_bind = bool(register)
        self._row: Dict[str, Any] = {}
        self._sync_object_id()
        self._sync_attrs(source if callable(source) else None)
        if callable(source):
            self._bind_func(source)
        elif isinstance(source, str):
            self._bind_template(source, tr_keys=tr_keys)
        elif source is not None:
            raise TypeError("Prompt source must be a callable, a template string, or None")

    def _row_id(self) -> str:
        if not self.name:
            raise ValueError("Prompt name is required")
        self._sync_object_id()
        return str(self.object_id)

    def _sync_object_id(self) -> None:
        if self.name:
            self.object_id = prompt_row_id(self.name, self.version)

    def _sync_attrs(self, fn: Optional[Callable[..., Any]] = None) -> None:
        self.__name__ = getattr(fn, "__name__", self.name or "prompt")
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__module__ = getattr(fn, "__module__", __name__)
        self.__doc__ = getattr(fn, "__doc__", None) or self.description
        self.__annotations__ = dict(getattr(fn, "__annotations__", {}) or {}) if fn is not None else {}

    @property
    def __signature__(self) -> Any:
        """Return the callable signature used by ``inspect.signature``.

        Returns:
            Any: The bound callable signature, or ``**kwargs`` for an unbound prompt.
        """
        fn = self._func()
        if fn is not None:
            return inspect.signature(fn)
        return inspect.Signature([inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)])

    def _bind_func(self, fn: Callable[..., Any]) -> "Prompt":
        if not callable(fn):
            raise TypeError("Prompt can only bind callable objects")
        if self.name is None:
            self.name = check_prompt_name(getattr(fn, "__name__", "prompt"))
            self.dicts = [self.name]
            self._sync_object_id()
        if not self.description:
            self.description = str(getattr(fn, "__doc__", "") or "").strip()
        self._capture_options.update(
            {
                "name": self.name,
                "namespace": "prompt",
                "version": self.version,
                "description": self.description,
                "tags": list(self.tags),
                "layers": ("source",),
            }
        )
        capsule = Capsule(fn, resolver=self._resolver, config=self._config, **self._capture_options)
        self._adopt_capsule(capsule, fn)
        self._sync_object_id()
        self._sync_attrs(fn)
        if self._register_on_bind:
            self.register(ws=self.ws)
        return self

    def _bind_template(self, template: str, tr_keys: Optional[List[str]] = None) -> "Prompt":
        capsule = Capsule.from_code(
            _template_code(self.name or "prompt", str(template), list(tr_keys or [])),
            fn_name="render",
            name=self.name or "prompt",
            namespace="prompt",
            version=self.version,
            description=self.description or "Render a string prompt template.",
            tags=self.tags,
            layers=("source",),
            resolver=self._resolver,
            config=self._config,
        )
        self._adopt_capsule(capsule, capsule.to_func())
        if self._register_on_bind:
            self.register(ws=self.ws)
        return self

    def _adopt_capsule(self, capsule: Capsule, fn: Optional[Callable[..., Any]] = None) -> None:
        self.manifest = capsule.manifest
        self._callable = fn
        self._sync_object_id()
        if fn is not None:
            self._sync_attrs(fn)

    def _tr(self, lang: Optional[str] = None) -> Callable[[Any], Any]:
        selected_lang = self._resolve_lang(lang)

        def translate(text: Any) -> str:
            return self.tr.get(str(text), selected_lang)

        return translate

    def _resolve_lang(self, lang: Optional[str]) -> str:
        config = prompt_config(self.ws)
        return str(
            lang or self.lang or config.get("heavenbase.prompt.lang", default=None) or self.main_lang or config.get("heavenbase.prompt.main_lang", default="en")
        )

    def _bind_workspace(self, workspace: Any) -> None:
        """Bind workspace-local prompt defaults that were omitted at construction."""
        config = prompt_config(workspace)
        self._bind_authority(
            resolver=getattr(workspace, "module_resolver", None),
            config=config,
        )
        if self._main_lang_defaulted:
            self.main_lang = str(config.get("heavenbase.prompt.main_lang", default="en") or "en")
        if self._elicit_defaulted:
            self.elicit = str(config.get("heavenbase.prompt.elicit", default="none") or "none")
        self.ws = workspace
        self.tr.ws = workspace

    def _render_func(self, args: tuple[Any, ...], kwargs: Dict[str, Any], lang: str) -> Any:
        fn = self._func()
        if fn is None:
            raise RuntimeError(f"Prompt {self.name!r} is not bound to a function")
        call_kwargs = dict(kwargs)
        if "tr" not in call_kwargs and "tr" in inspect.signature(fn).parameters:
            call_kwargs["tr"] = self._tr(lang)
        return fn(*args, **call_kwargs)

    def _func(self) -> Optional[Callable[..., Any]]:
        if self._callable is None and self.manifest is not None:
            self._callable = Capsule.from_manifest(
                self.manifest,
                resolver=self._resolver,
                config=self._config,
            ).to_func()
            self._sync_attrs(self._callable)
        return self._callable

    def _ensure_capsule(self) -> Dict[str, Any]:
        if self.manifest is not None:
            return self.manifest.to_dict()
        if self._callable is None:
            raise RuntimeError(f"Prompt {self.name!r} is not bound to a function")
        capsule = Capsule(
            self._callable,
            name=self.name,
            namespace="prompt",
            version=self.version,
            description=self.description,
            tags=self.tags,
            layers=("source",),
            resolver=self._resolver,
            config=self._config,
        )
        self.manifest = capsule.manifest
        return self.manifest.to_dict()

    def to_manifest(self) -> Dict[str, Any]:
        """Return prompt-owned manifest metadata.

        Args:
            None.

        Returns:
            Dict[str, Any]: JSON-compatible prompt metadata stored with the ``sys-prompt`` row.
        """
        return {
            "object_id": self._row_id(),
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tags": list(self.tags),
            "dicts": list(self.dicts),
            "main_lang": self.main_lang,
            "elicit": self.elicit,
            "meta": dict(self.meta),
        }

    def register(self, ws: Any = None, overwrite: bool = True, reason: str = "") -> str:
        """Persist this prompt into a HeavenBase workspace.

        Args:
            ws (Any | None): Workspace to write to. Defaults to the bound or active workspace.
            overwrite (bool): Whether to replace an existing same-object_id row. Defaults to ``True``.
            reason (str): Optional revision note stored in the prompt manifest.

        Returns:
            str: Stable prompt row object_id.

        Warnings:
            Registration stores executable Capsule source in the prompt row. Review generated or captured code before
            sharing a workspace with less-trusted callers.
        """
        workspace = resolve_prompt_workspace(ws if ws is not None else self.ws)
        self._bind_workspace(workspace)
        ensure_prompt_entities(workspace)
        row_id = self._row_id()
        capsule = self._ensure_capsule()
        manifest = self.to_manifest()
        if reason:
            manifest["reason"] = reason
        now = time()
        existing = workspace.get(row_id, entity=type(self))
        created_at = float((existing or {}).get("created_at") or now)
        row = {
            "object_id": row_id,
            "name": self.name,
            "version": self.version,
            "status": "active",
            "description": self.description,
            "tags": list(self.tags),
            "manifest": manifest,
            "capsule": capsule,
            "dicts": list(self.dicts),
            "main_lang": self.main_lang,
            "fingerprints": {"manifest": sha256hash(manifest), "capsule": sha256hash(capsule)},
            "created_at": created_at,
            "updated_at": now,
        }
        workspace.upsert_many(type(self), [row], overwrite=overwrite)
        self._row = dict(row)
        return row_id

    def to_dict(self) -> Dict[str, Any]:
        """Return the persistent ``sys-prompt`` row for this prompt.

        Args:
            None.

        Returns:
            Dict[str, Any]: JSON-compatible row ready for ``workspace.upsert(hb.Prompt, row)``.
        """
        capsule = self._ensure_capsule()
        manifest = self.to_manifest()
        now = time()
        return {
            "object_id": self._row_id(),
            "name": self.name,
            "version": self.version,
            "status": "active",
            "description": self.description,
            "tags": list(self.tags),
            "manifest": manifest,
            "capsule": capsule,
            "dicts": list(self.dicts),
            "main_lang": self.main_lang,
            "fingerprints": {"manifest": sha256hash(manifest), "capsule": sha256hash(capsule)},
            "created_at": now,
            "updated_at": now,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, **values: Any) -> "Prompt":
        """Build a prompt object from a persisted ``sys-prompt`` row.

        Args:
            data (Dict[str, Any] | None): Prompt row returned by ``Prompt.describe(...)`` or
                ``workspace.get(..., entity=hb.Prompt)``. Defaults to ``None``.
            **values (Any): Additional row values merged over ``data``.

        Returns:
            Prompt: Prompt object with lazy Capsule restoration metadata attached.

        Warnings:
            Calling the returned prompt may restore and execute Capsule source code from the row.
            Only hydrate prompts from trusted workspaces.

        Examples:
            >>> row = Prompt("Hello, {name}", name="demo.hello").to_dict()
            >>> Prompt.from_dict(row)(name="Ada")
            'Hello, Ada'
        """
        row = dict(data or {})
        row.update(values)
        return cls._from_row(row)

    @classmethod
    def _from_row(cls, row: Dict[str, Any], *, ws: Any = None, lang: Optional[str] = None) -> "Prompt":
        manifest = dict(row.get("manifest") or {})
        name = str(row.get("name") or manifest.get("name") or "")
        config = prompt_config(ws)
        prompt = cls(
            name=name,
            version=str(row.get("version") or manifest.get("version") or "1"),
            ws=ws,
            dicts=list(row.get("dicts") or manifest.get("dicts") or ([name] if name else [])),
            main_lang=str(row.get("main_lang") or manifest.get("main_lang") or config.get("heavenbase.prompt.main_lang", default="en")),
            lang=lang,
            elicit=str(manifest.get("elicit") or config.get("heavenbase.prompt.elicit", default="none") or "none"),
            description=str(row.get("description") or manifest.get("description") or ""),
            tags=list(row.get("tags") or manifest.get("tags") or []),
            **dict(manifest.get("meta") or {}),
        )
        capsule_manifest = dict(row.get("capsule") or {})
        if capsule_manifest:
            prompt.manifest = CapsuleManifest.from_dict(capsule_manifest)
        prompt._row = dict(row)
        return prompt

    @classmethod
    def load(cls, ref: str, ws: Any = None, version: Optional[str] = None, lang: Optional[str] = None) -> "Prompt":
        """Load one active prompt by dotted name, ``name:version``, or row object_id.

        Args:
            ref (str): Dotted name, compact ``name:version`` reference, or stable row object_id.
            ws (Any | None): Workspace to read from. Defaults to the active workspace.
            version (str | None): Explicit version override when ``ref`` is a name.
            lang (str | None): Runtime language bound to the loaded prompt.

        Returns:
            Prompt: Loaded prompt with lazy Capsule restoration.

        Raises:
            KeyError: If no active prompt row matches the reference.

        Warnings:
            Calling the returned prompt may restore and execute Capsule source code from the selected row.
        """
        workspace = resolve_prompt_workspace(ws)
        ensure_prompt_entities(workspace)
        row = _row_for_ref(workspace, parse_prompt_ref(ref), version=version)
        if row is None or row.get("status") != "active":
            raise KeyError(f"No active prompt found for {ref!r}")
        return cls._from_row(row, ws=workspace, lang=lang)

    @classmethod
    def list(cls, prefix: Optional[str] = None, status: Optional[str] = "active", ws: Any = None) -> List[Dict[str, Any]]:
        """List prompt rows in a workspace.

        Args:
            prefix (str | None): Optional dotted-name prefix filter.
            status (str | None): Optional row status filter. Pass ``None`` to include tombstoned rows.
            ws (Any | None): Workspace to query. Defaults to the active workspace.

        Returns:
            List[Dict[str, Any]]: Prompt rows sorted by name, version, and update timestamp.
        """
        workspace = resolve_prompt_workspace(ws)
        ensure_prompt_entities(workspace)
        rows = workspace.query(cls).execute().rows()
        if status is not None:
            rows = [row for row in rows if row.get("status") == status]
        if prefix:
            rows = [row for row in rows if str(row.get("name") or "").startswith(prefix)]
        return sorted(rows, key=lambda row: (str(row.get("name") or ""), _version_key(row), float(row.get("updated_at") or 0)))

    @classmethod
    def describe(cls, ref: str, ws: Any = None, version: Optional[str] = None) -> Dict[str, Any]:
        """Return one prompt row by reference.

        Args:
            ref (str): Prompt name, ``name:version``, or row object_id.
            ws (Any | None): Workspace to read from. Defaults to the active workspace.
            version (str | None): Explicit version override when ``ref`` is a name.

        Returns:
            Dict[str, Any]: Matching prompt row.

        Raises:
            KeyError: If no prompt row matches the reference.
        """
        workspace = resolve_prompt_workspace(ws)
        ensure_prompt_entities(workspace)
        row = _row_for_ref(workspace, parse_prompt_ref(ref), version=version)
        if row is None:
            raise KeyError(f"No prompt found for {ref!r}")
        return row

    @classmethod
    def versions(cls, name: str, ws: Any = None) -> List[str]:
        """Return versions known for a prompt name.

        Args:
            name (str): Dotted prompt name.
            ws (Any | None): Workspace to query. Defaults to the active workspace.

        Returns:
            List[str]: Versions sorted by numeric version components when possible.
        """
        clean_name = check_prompt_name(name)
        rows = [row for row in cls.list(status=None, ws=ws) if row.get("name") == clean_name]
        return [str(row.get("version") or "1") for row in sorted(rows, key=_version_key)]

    @classmethod
    def delete(cls, ref: str, version: Optional[str] = None, tombstone: bool = True, ws: Any = None) -> bool:
        """Delete or tombstone a prompt row.

        Args:
            ref (str): Prompt name, ``name:version``, or row object_id.
            version (str | None): Explicit version override when ``ref`` is a name.
            tombstone (bool): Mark the row deleted instead of removing it physically. Defaults to ``True``.
            ws (Any | None): Workspace to mutate. Defaults to the active workspace.

        Returns:
            bool: Whether a row was found and removed or tombstoned.
        """
        workspace = resolve_prompt_workspace(ws)
        ensure_prompt_entities(workspace)
        row = _row_for_ref(workspace, parse_prompt_ref(ref), version=version)
        if row is None:
            return False
        row_id = row.get("object_id")
        if tombstone:
            row = dict(row)
            row["object_id"] = row_id
            row["status"] = "deleted"
            row["updated_at"] = time()
            workspace.set(row, entity=cls)
            return True
        return workspace.delete((cls, row_id))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Bind a decorator function or call the prompt.

        Args:
            *args (Any): Function/template positional arguments, or one callable when used as a decorator.
            **kwargs (Any): Function/template keyword arguments. ``lang`` is consumed as the runtime language override.

        Returns:
            Any: The wrapped callable return value.

        Raises:
            RuntimeError: If the prompt has no bound callable or Capsule manifest.

        Examples:
            >>> Prompt("Hello, {name}", name="demo.hello")(name="Ada")
            'Hello, Ada'
        """
        if self._callable is None and self.manifest is None and len(args) == 1 and callable(args[0]) and not kwargs:
            return self._bind_func(args[0])
        return self._render_func(args, kwargs, self._resolve_lang(kwargs.pop("lang", None)))

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Render the prompt with the same translation behavior as ``prompt(...)``.

        Args:
            *args (Any): Positional render arguments forwarded to the bound
                prompt callable or template.
            **kwargs (Any): Keyword render arguments forwarded to the bound
                prompt callable or template.

        Returns:
            Any: Rendered prompt output.
        """
        return self(*args, **kwargs)


def _version_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    version = str(row.get("version") or "1")
    parts = version.split(".")
    if parts and all(part.isdigit() for part in parts):
        return (1, [int(part) for part in parts], "", float(row.get("updated_at") or 0))
    return (0, [], version, float(row.get("updated_at") or 0))


def _latest(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return sorted(rows, key=_version_key)[-1] if rows else None


def _row_for_ref(workspace: Any, parsed: tuple[Optional[str], Optional[str], Optional[str]], version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    name, ref_version, row_id = parsed
    if row_id:
        return workspace.get(row_id, entity=Prompt)
    selected_version = version or ref_version
    if selected_version:
        return workspace.get(prompt_row_id(name, str(selected_version)), entity=Prompt)
    rows = [row for row in workspace.query(Prompt).where({"name": name}).where({"status": "active"}).execute().rows()]
    return _latest(rows)


def ensure_prompt_entities(ws: Any) -> Any:
    """Ensure prompt-extension entities exist in a workspace.

    Args:
        ws (Any): HeavenBase workspace.

    Returns:
        Any: The same workspace after enabling the built-in ``prompt`` extension.
    """
    enable_extension = getattr(ws, "enable_extension", None)
    if callable(enable_extension):
        enable_extension("prompt")
        return ws
    ws.register(Prompt)
    ws.register(Translation)
    return ws
