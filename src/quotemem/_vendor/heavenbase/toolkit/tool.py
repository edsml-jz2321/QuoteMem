"""Single-tool runtime and serializer behavior for system Toolkits."""

from __future__ import annotations

__all__ = ["Tool"]

import inspect
import threading

from heavenbase.capsule import Capsule
from heavenbase.modules import ModuleResolver
from heavenbase.utils import Any, Callable, Dict, List, Mapping, Optional, dataclass, field as dcfield
from heavenbase.utils.config import ConfigManager

from .errors import AsyncToolRequired
from .serializers import jstr_serializer


def _jstr_serializer_capsule(capsule: Capsule) -> Capsule:
    """Return the shared compact-JSON serializer Capsule.

    Args:
        capsule (Capsule): Execution Capsule whose namespace, version, and
            resolver authority should be reused.

    Returns:
        Capsule: Compact-JSON serializer bound to the same resolver.
    """
    return Capsule(
        jstr_serializer,
        name="jstr_serializer",
        namespace=capsule.manifest.namespace,
        version=capsule.manifest.version,
        description="Serialize top-level dict/list with compact JSON; all other values with str().",
        resolver=capsule._resolver,
        config=capsule._config,
    )


@dataclass(init=False)
class Tool:
    """Function-like runtime view over one execution Capsule.

    ``Tool`` accepts an existing Capsule, captures a callable directly, or
    acts as either a bare or configured decorator.

    Warnings:
        Running a Tool runs the underlying Capsule. Only expose Tool objects to agents when the Capsule source and input
        schema are trusted for that agent boundary.

    Examples:
        >>> @Tool
        ... def add(left: int, right: int) -> int:
        ...     return left + right
        >>> add(left=2, right=3)
        5
        >>> @Tool(name="multiply")
        ... def mul(left: int, right: int) -> int:
        ...     return left * right
        >>> mul(left=2, right=3)
        6
    """

    name: str
    capsule: Capsule | None
    description: str = ""
    input_schema: Dict[str, Any] = dcfield(default_factory=dict)
    output_schema: Dict[str, Any] = dcfield(default_factory=dict)
    annotations: Dict[str, Any] = dcfield(default_factory=dict)
    metadata: Dict[str, Any] = dcfield(default_factory=dict)
    serializer: Any = None
    _default_serializer: Capsule | None = dcfield(default=None, init=False, repr=False, compare=False)
    _serializer_config: Any = dcfield(default=None, init=False, repr=False, compare=False)
    _serializer_lock: Any = dcfield(default_factory=threading.RLock, init=False, repr=False, compare=False)

    def __init__(
        self,
        source: Callable[..., Any] | Capsule | None = None,
        description: str = "",
        input_schema: Optional[Mapping[str, Any]] = None,
        output_schema: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        serializer: Any = None,
        *,
        annotations: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        version: str = "1",
        tags: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
        layer_options: Optional[Mapping[str, Any]] = None,
        include_cloudpickle: bool = False,
        prefer_import: bool = True,
        resolver: ModuleResolver | None = None,
        config: ConfigManager | None = None,
    ) -> None:
        """Create a Tool from a callable, Capsule, or decorator options.

        Args:
            source: Callable or Capsule to wrap. `None` creates an unbound
                decorator.
            description: Public tool description. Defaults to Capsule metadata.
            input_schema: Optional input-schema override.
            output_schema: Optional output-schema override.
            metadata: Additional JSON-compatible tool metadata.
            serializer: Optional callable or Capsule output serializer.
            annotations: Optional MCP ``ToolAnnotations`` mapping using the
                protocol field names such as ``readOnlyHint``. The mapping is
                retained by Toolkit persistence and validated at MCP export.
            name: Public Tool and captured Capsule name.
            namespace: Capsule namespace.
            version: Capsule version. Defaults to ``"1"``.
            tags: Capsule search tags.
            layers: Optional Capsule capture layer order.
            layer_options: Optional per-layer capture options.
            include_cloudpickle: Whether to include a trusted-local binary
                restore layer. Defaults to ``False``.
            prefer_import: Whether import-path restore is preferred. Defaults
                to ``True``.
            resolver: Module authority used for Capsule capture and restore.
            config: Config authority paired with ``resolver``.

        Raises:
            TypeError: If `source` is not callable or a Capsule.

        Returns:
            None: This initializer stores the Tool runtime state.
        """
        if source is not None and not isinstance(source, Capsule) and not callable(source):
            raise TypeError("Tool source must be a callable, Capsule, or None")

        self.name = str(name or "")
        self.capsule = None
        self.description = str(description or "")
        self.input_schema = dict(input_schema or {})
        self.output_schema = dict(output_schema or {})
        self.annotations = dict(annotations or {})
        self.metadata = dict(metadata or {})
        self.serializer = None
        self._default_serializer = None
        self._serializer_config = config
        self._serializer_lock = threading.RLock()
        self._capture_options = {
            "name": name,
            "namespace": namespace,
            "version": version,
            "description": description or None,
            "tags": tags,
            "layers": layers,
            "layer_options": dict(layer_options or {}),
            "include_cloudpickle": include_cloudpickle,
            "prefer_import": prefer_import,
            "resolver": resolver,
            "config": config,
        }
        self._pending_serializer = serializer
        if isinstance(source, Capsule):
            self._bind_capsule(source)
        elif callable(source):
            self._bind_func(source)
        else:
            self._sync_func_attrs()

    def __post_init__(self) -> None:
        """Normalize public metadata after a Capsule is bound."""
        capsule = self._require_capsule()
        if not self.input_schema:
            self.input_schema = dict(capsule.manifest.input_schema)
        if not self.output_schema:
            self.output_schema = dict(capsule.manifest.output_schema)
        if not self.description:
            self.description = capsule.manifest.description
        pending_serializer = self._pending_serializer
        self._pending_serializer = None
        if pending_serializer is not None:
            self.set_serializer(pending_serializer, config=self._serializer_config)
        self._sync_func_attrs(capsule)

    def _display_name(self) -> str:
        """Return a stable name for bound and unbound debug messages."""
        return self.name or str(self._capture_options.get("name") or "tool")

    def _require_capsule(self) -> Capsule:
        """Return the bound Capsule or fail with an actionable error."""
        if self.capsule is None:
            raise RuntimeError(f"Tool {self._display_name()!r} is not bound to a function")
        return self.capsule

    def _sync_func_attrs(self, fn: Optional[Callable[..., Any]] = None) -> None:
        """Expose wrapped-function metadata on the Tool object."""
        self.__name__ = self.name or getattr(fn, "__name__", "tool")
        self.__qualname__ = str(getattr(fn, "__qualname__", self.__name__))
        self.__module__ = str(getattr(fn, "__module__", ""))
        self.__doc__ = self.description or getattr(fn, "__doc__", None)
        self.__annotations__ = dict(getattr(fn, "__annotations__", {}) or {}) if fn is not None else {}

    def _bind_capsule(self, capsule: Capsule) -> "Tool":
        """Bind one existing Capsule and normalize public Tool metadata."""
        self.capsule = capsule
        self.name = self.name or capsule.manifest.name
        self.__post_init__()
        return self

    def _bind_func(self, fn: Callable[..., Any]) -> "Tool":
        """Capture and bind a callable supplied directly or by decoration."""
        if not callable(fn):
            raise TypeError("Tool can only bind callable objects")
        options = {key: value for key, value in self._capture_options.items() if value is not None}
        return self._bind_capsule(Capsule(fn, **options))

    def _serializer_capsule(self) -> Capsule:
        if isinstance(self.serializer, Capsule):
            return self.serializer
        if self._default_serializer is None:
            with self._serializer_lock:
                if self._default_serializer is None:
                    self._default_serializer = _jstr_serializer_capsule(self._require_capsule())
        return self._default_serializer

    def _bind_serializer_config(self, config: Any | None) -> None:
        """Bind the config authority supplied by the owning Toolkit."""
        self._serializer_config = config

    def _bind_authority(self, resolver: Any, config: Any) -> None:
        """Bind every execution and serializer Capsule to one Toolkit authority."""
        self._require_capsule()._bind_authority(resolver=resolver, config=config)
        for serializer in (self.serializer, self._default_serializer):
            if isinstance(serializer, Capsule):
                serializer._bind_authority(resolver=resolver, config=config)
        self._bind_serializer_config(config)

    def _serialize(self, output: Any) -> Any:
        """Run the serializer with config injection when its contract allows it."""
        serializer = self._serializer_capsule().to_func()
        if self._serializer_config is not None:
            try:
                inspect.signature(serializer).bind(output, config=self._serializer_config)
            except (TypeError, ValueError):
                pass
            else:
                return serializer(output, config=self._serializer_config)
        return serializer(output)

    def set_serializer(
        self,
        serializer: Any = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
        version: Optional[str] = None,
        layers: Optional[List[str]] = None,
        layer_options: Optional[Mapping[str, Any]] = None,
        include_cloudpickle: bool = False,
        prefer_import: bool = True,
        config: Any | None = None,
    ) -> "Tool":
        """Set the serializer Capsule for this Tool.

        Args:
            serializer (Any): Callable or Capsule that accepts the raw output
                and returns text. Config-bound Toolkits additionally pass the
                keyword-only ``config`` authority when the callable accepts
                it. ``None`` restores the default :func:`jstr_serializer`
                behavior.
            name (Optional[str]): Serializer Capsule name. Defaults to ``"<tool>_serializer"`` for custom serializer callables.
            description (Optional[str]): Serializer Capsule description. Defaults to a generated description.
            namespace (Optional[str]): Serializer Capsule namespace. Defaults to the main Capsule namespace.
            version (Optional[str]): Serializer Capsule version. Defaults to the main Capsule version.
            layers (Optional[List[str]]): Optional Capsule layer order for serializer capture. Defaults to ``None``.
            layer_options (Optional[Mapping[str, Any]]): Optional per-layer capture options. Defaults to ``None``.
            include_cloudpickle (bool): Whether to store a trusted-local binary fallback. Defaults to ``False``.
            prefer_import (bool): Whether import-path restore should be preferred when available. Defaults to ``True``.
            config: Config authority passed to the serializer at execution.

        Returns:
            Tool: This Tool for chaining.
        """
        self._bind_serializer_config(config)
        if isinstance(serializer, Capsule):
            self.serializer = serializer
            return self
        capsule = self._require_capsule()
        capsule_namespace = namespace or capsule.manifest.namespace
        capsule_version = version or capsule.manifest.version
        if serializer is None:
            self.serializer = None
            return self
        fn = serializer
        if not callable(fn):
            raise TypeError(f"Tool serializer for {self.name!r} must be a callable or Capsule")
        self.serializer = Capsule(
            fn,
            name=name or f"{self.name}_serializer",
            namespace=capsule_namespace,
            version=capsule_version,
            description=description or f"Serialize {self.name} tool output.",
            layers=layers,
            layer_options=dict(layer_options or {}),
            include_cloudpickle=include_cloudpickle,
            prefer_import=prefer_import,
            resolver=capsule._resolver,
            config=capsule._config,
        )
        return self

    @property
    def __signature__(self) -> Any:
        """Return the wrapped callable signature for ``inspect.signature``."""
        if self.capsule is None:
            return inspect.Signature([inspect.Parameter("fn", inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        return inspect.signature(self.capsule)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Bind a decorator function or call the wrapped Tool."""
        if self.capsule is None and len(args) == 1 and callable(args[0]) and not kwargs:
            return self._bind_func(args[0])
        return self.run(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run this tool locally.

        Args:
            *args (Any): Positional arguments passed to the underlying Capsule.
            **kwargs (Any): Keyword arguments passed to the underlying Capsule.

        Returns:
            Any: Native Python return value from the Capsule.
        """
        return self._require_capsule().run(*args, **kwargs)

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        """Run this tool locally and await awaitable return values.

        Args:
            *args (Any): Positional arguments passed to the underlying Capsule.
            **kwargs (Any): Keyword arguments passed to the underlying Capsule.

        Returns:
            Any: Native Python return value from the Capsule.
        """
        return await self._require_capsule().arun(*args, **kwargs)

    def to_str(self, output: Any) -> str:
        """Convert a raw tool output through this Tool's serializer Capsule.

        Args:
            output (Any): Native Python value returned by ``run``.

        Returns:
            str: Serialized tool output for agent and MCP boundaries.

        Raises:
            AsyncToolRequired: If the serializer returns an awaitable value.
            TypeError: If the serializer returns a non-string value.
        """
        text = self._serialize(output)
        if inspect.isawaitable(text):
            raise AsyncToolRequired("async tool serializer needs ato_str")
        if not isinstance(text, str):
            raise TypeError(f"Tool serializer for {self.name!r} must return str, got {type(text).__name__}")
        return text

    async def ato_str(self, output: Any) -> str:
        """Convert a raw tool output through this Tool's serializer Capsule and await async serializers.

        Args:
            output (Any): Native Python value returned by ``run`` or ``arun``.

        Returns:
            str: Serialized tool output for agent and MCP boundaries.

        Raises:
            TypeError: If the serializer returns a non-string value.
        """
        text = self._serialize(output)
        if inspect.isawaitable(text):
            text = await text
        if not isinstance(text, str):
            raise TypeError(f"Tool serializer for {self.name!r} must return str, got {type(text).__name__}")
        return text

    def run_to_str(self, *args: Any, **kwargs: Any) -> str:
        """Run this tool and return its serialized string output.

        Args:
            *args (Any): Positional arguments passed to the underlying Capsule.
            **kwargs (Any): Keyword arguments passed to the underlying Capsule.

        Returns:
            str: Serialized tool output for agent, MCP, and LLM boundaries.

        Raises:
            AsyncToolRequired: If the Capsule returns an awaitable value. Use ``arun_to_str`` instead.
        """
        value = self.run(*args, **kwargs)
        if inspect.isawaitable(value):
            raise AsyncToolRequired("async tool execution needs arun_to_str")
        return self.to_str(value)

    async def arun_to_str(self, *args: Any, **kwargs: Any) -> str:
        """Run this tool asynchronously and return its serialized string output.

        Args:
            *args (Any): Positional arguments passed to the underlying Capsule.
            **kwargs (Any): Keyword arguments passed to the underlying Capsule.

        Returns:
            str: Serialized tool output for agent, MCP, and LLM boundaries.
        """
        return await self.ato_str(await self.arun(*args, **kwargs))

    def to_openai_tool(self) -> Dict[str, Any]:
        """Return an OpenAI-compatible function tool schema.

        Args:
            None.

        Returns:
            Dict[str, Any]: Function-tool schema with normalized object parameters.
        """
        schema = dict(self.input_schema or {})
        if schema.get("type") != "object" or "properties" not in schema:
            schema = {"type": "object", "properties": schema or {}, "required": []}
        schema.setdefault("properties", {})
        schema.setdefault("required", [])
        return {"type": "function", "function": {"name": self.name, "description": self.description or "", "parameters": schema}}
