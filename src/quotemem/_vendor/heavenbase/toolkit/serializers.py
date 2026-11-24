"""Registered MCP tool output serializers."""

from __future__ import annotations

__all__ = ["SerializerFn", "ToolSerializer", "jstr_serializer", "markdown_serializer"]

import inspect

from heavenbase.utils import Any, Callable, Mapping, Sequence, dataclass, dumps_jstr, raise_mismatch
from heavenbase.utils.config import CM_HVNB
from heavenbase.utils.registry import named_key

from heavenbase.extensions.contributions import eligible_contributions
from heavenbase.modules import (
    ModuleResolver,
    ModuleService,
    capsule_executable_ref,
    module_resolver,
    python_executable_ref,
)
from heavenbase.modules.artifacts import SourceUnavailable
from heavenbase.modules.base import ResolverCache

SerializerFn = Callable[..., str]

_SERIALIZER_CACHE = ResolverCache["ToolSerializer"]()


def _omitted(config: Any | None) -> Any:
    """Return ``config``, or the default Context's config when it is omitted."""
    if config is not None:
        return config
    from heavenbase.context.authority import omitted_config

    return omitted_config()


@dataclass(frozen=True)
class ToolSerializer:
    """Name one callable that renders raw tool output as text.

    Args:
        name (str): Serializer identifier referenced by `ProfileSpec.serializer`.
        serialize (SerializerFn): Callable accepting ``(output, *, config)``
            and returning text.
    """

    name: str
    serialize: SerializerFn

    def __post_init__(self) -> None:
        """Normalize the name and validate the serializer callable.

        Args:
            None.

        Returns:
            None: This initializer stores a canonical name.

        Raises:
            TypeError: If `serialize` is not callable or does not accept the
                config-injected serializer protocol.
        """
        name = named_key(self.name, "MCP serializer name")
        if not callable(self.serialize):
            raise TypeError(f"MCP serializer {name!r} must be callable")
        try:
            inspect.signature(self.serialize).bind(object(), config=CM_HVNB)
        except (TypeError, ValueError) as error:
            raise TypeError(f"MCP serializer {name!r} must accept (output, *, config)") from error
        object.__setattr__(self, "name", name)

    def __call__(self, output: Any, *, config: Any | None = None) -> str:
        """Render one raw tool result.

        Args:
            output (Any): Raw value returned by a tool.
            config: Config authority used by serializer policy. When omitted,
                the process default is selected.

        Returns:
            str: Serialized tool output.

        Raises:
            TypeError: If the serializer does not return text.
        """
        text = self.serialize(output, config=_omitted(config))
        if not isinstance(text, str):
            raise TypeError(f"MCP serializer {self.name!r} must return str, got {type(text).__name__}")
        return text

    def register(
        self,
        *,
        replace: bool = False,
        resolver: ModuleResolver | None = None,
    ) -> "ToolSerializer":
        """Publish this serializer through the central module Registry.

        Args:
            replace (bool): Whether to replace an existing serializer.
            resolver (ModuleResolver | None): Writable Context-owned module
                resolver. When omitted, the default Context is selected.

        Returns:
            ToolSerializer: This registered descriptor.

        Raises:
            ValueError: If the name already exists and `replace` is `False`.
        """
        chosen = module_resolver(resolver)
        if not isinstance(chosen, ModuleService):
            raise TypeError("MCP serializer registration requires a writable ModuleService")
        service = chosen
        found = service.inspect("serializer", self.name)
        if found:
            existing = _serializer_from_row(found[0], service)
            if existing == self and replace:
                return existing
            if not replace:
                raise ValueError(f"MCP serializer {self.name!r} is already registered")
            coordinate = found[0]["record"]["meta"]["installation"]["coordinate"]
            if coordinate != f"programmatic/serializer/{self.name}":
                raise ValueError(f"MCP serializer {self.name!r} belongs to {coordinate!r} and cannot be replaced")
        reference = _serializer_reference(self.serialize, service)
        service.register(
            "serializer",
            self.name,
            {
                "name": self.name,
                "desc": _doc_summary(self.serialize),
                "source": "inline",
                "target": "definition",
                "active": True,
                "meta": {
                    "schema_version": 1,
                    "tags": {"toolkit": True, "mcp": True},
                    "definition": {"serialize": reference},
                },
            },
        )
        return _load_serializer(self.name, service)

    @classmethod
    def load(
        cls,
        name: str,
        *,
        aliases: Mapping[str, SerializerFn] | None = None,
        resolver: ModuleResolver | None = None,
        workspace: Any | None = None,
    ) -> "ToolSerializer":
        """Load a registered serializer or workspace-local alias.

        Args:
            name (str): Serializer identifier.
            aliases (Mapping[str, SerializerFn] | None): Optional workspace-local
                serializer callables.
            resolver (ModuleResolver | None): Context-owned module resolver.
                When omitted, the default Context is selected.
            workspace: Optional workspace whose active extensions constrain
                Registry contributions.

        Returns:
            ToolSerializer: Resolved serializer descriptor.

        Raises:
            ValueError: If `name` is unknown.
        """
        if aliases and name in aliases:
            return cls(name, aliases[name])
        chosen = module_resolver(resolver)
        try:
            serializer = _load_serializer(name, chosen, workspace=workspace)
        except KeyError:
            known = sorted((*_serializer_names(chosen, workspace=workspace), *(aliases or {})))
            raise_mismatch(known, name, name="MCP profile serializer")
        return serializer

    @classmethod
    def snapshot(
        cls,
        extra: Mapping[str, SerializerFn] | None = None,
        *,
        resolver: ModuleResolver | None = None,
        workspace: Any | None = None,
    ) -> dict[str, SerializerFn]:
        """Return registered serializer callables with optional local aliases.

        Args:
            extra (Mapping[str, SerializerFn] | None): Optional aliases layered
                over the process registry snapshot.
            resolver (ModuleResolver | None): Context-owned module resolver.
                When omitted, the default Context is selected.
            workspace: Optional workspace whose active extensions constrain
                Registry contributions.

        Returns:
            dict[str, SerializerFn]: Detached name-to-callable mapping.
        """
        chosen = module_resolver(resolver)
        serializers = {name: _load_serializer(name, chosen, workspace=workspace).serialize for name in _serializer_names(chosen, workspace=workspace)}
        if extra:
            serializers.update(dict(extra))
        return serializers

    def to_ref(self) -> str:
        """Return the stable import-style reference for this serializer.

        Args:
            None.

        Returns:
            str: Module-qualified callable name, or `repr` when unavailable.
        """
        module = getattr(self.serialize, "__module__", "")
        qualname = getattr(self.serialize, "__qualname__", getattr(self.serialize, "__name__", ""))
        return f"{module}.{qualname}" if module and qualname else repr(self.serialize)


def jstr_serializer(output: Any, *, config: Any | None = None) -> str:
    """Serialize tool output as compact JSON-compatible text.

    Args:
        output (Any): Raw tool output.
        config: Config authority reserved for uniform serializer injection.

    Returns:
        str: Compact JSON text for structured values and string text otherwise.
    """
    return dumps_jstr(output, compact=True)


def markdown_serializer(output: Any, *, config: Any | None = None) -> str:
    """Serialize frame-like tool output as a capped Markdown table.

    Args:
        output (Any): Raw tool output.
        config: Config authority supplying the table row limit. When omitted,
            the process default is selected.

    Returns:
        str: Markdown table when supported, otherwise compact JSON text.
    """
    table = _markdown_table(output, config=_omitted(config))
    return table if table is not None else jstr_serializer(output, config=config)


def _markdown_table(output: Any, *, config: Any) -> str | None:
    if not isinstance(output, Mapping):
        return None
    rows = output.get("rows")
    if isinstance(rows, list) and rows:
        return _rows_markdown(rows, config=config, withheld=_withheld_rows(output, len(rows)))
    row = output.get("row")
    if isinstance(row, Mapping) and row:
        return _rows_markdown([row], config=config)
    entities = output.get("entities")
    if isinstance(entities, list) and entities:
        return "\n".join(f"- {item}" for item in entities)
    return None


def _withheld_rows(output: Mapping[str, Any], returned: int) -> int:
    """Return how many rows the engine withheld before this payload was built.

    A payload that paginated carries its own pre-limit total, so the rendered
    table can state what the caller is not seeing instead of presenting a page
    as the whole result.
    """
    if not output.get("truncated"):
        return 0
    total = output.get("total")
    return max(0, int(total) - returned) if isinstance(total, int) else 0


def _rows_markdown(rows: Sequence[Mapping[str, Any]], *, config: Any, withheld: int = 0) -> str:
    max_rows = int(config.get("heavenbase.mcp.display.max_rows", default=50) or 50)
    shown = list(rows[:max_rows])
    columns = _column_names(shown)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_cell_text(row.get(col)) for col in columns) + " |" for row in shown]
    lines = [header, divider, *body]
    # Two independent cuts can hide rows: this renderer's display cap, and the
    # engine's own pagination. Both are stated, so a table never reads as whole.
    hidden = (len(rows) - max_rows if len(rows) > max_rows else 0) + withheld
    if hidden:
        lines.append(f"\n... {hidden} more row(s) truncated")
    return "\n".join(lines)


def _column_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            text = str(key)
            if text not in seen:
                seen.add(text)
                names.append(text)
    return names or ["value"]


def _cell_text(value: Any, *, max_chars: int = 120) -> str:
    if value is None:
        return ""
    text = dumps_jstr(value, compact=True) if isinstance(value, (dict, list)) else str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text


def _serializer_names(
    resolver: ModuleResolver,
    *,
    workspace: Any | None = None,
) -> tuple[str, ...]:
    """Return installed or workspace-eligible serializer identifiers."""
    rows = eligible_contributions(resolver.inspect("serializer"), workspace)
    return tuple(str(row["identifier"]) for row in rows)


def _available_serializers(
    resolver: ModuleResolver,
    *,
    workspace: Any | None = None,
) -> dict[str, SerializerFn]:
    """Return the serializer callables whose source can be loaded.

    For callers that match a callable against every registered serializer: one
    whose source is gone, such as a Capsule a drop removed, has no callable to
    match, so it is left out instead of failing the others.
    """
    available: dict[str, SerializerFn] = {}
    for name in _serializer_names(resolver, workspace=workspace):
        try:
            available[name] = _load_serializer(name, resolver, workspace=workspace).serialize
        except SourceUnavailable:
            continue
    return available


def _load_serializer(
    name: str,
    resolver: ModuleResolver,
    *,
    workspace: Any | None = None,
) -> ToolSerializer:
    """Materialize one installed or workspace-eligible serializer."""
    key = named_key(name, "MCP serializer name")
    rows = eligible_contributions(resolver.inspect("serializer", key), workspace)
    if not rows:
        raise KeyError(key)
    return _serializer_from_row(rows[0], resolver)


def _serializer_from_row(row: Mapping[str, Any], resolver: ModuleResolver) -> ToolSerializer:
    """Materialize and cache one serializer inspection row."""

    def build(current_row: Mapping[str, Any]) -> ToolSerializer:
        identifier = str(current_row["identifier"])
        raw = resolver.resolve("serializer", identifier)
        if not isinstance(raw, Mapping) or set(raw) != {"serialize"}:
            raise TypeError(f"MCP serializer {identifier!r} must define one serialize locator")
        reference = raw.get("serialize")
        if not isinstance(reference, Mapping):
            raise TypeError(f"MCP serializer {identifier!r} serialize must be an executable reference")
        return ToolSerializer(
            identifier,
            resolver.resolve_executable(reference, owner=current_row),
        )

    return _SERIALIZER_CACHE.get(resolver, row, build)


def _doc_summary(fn: Any) -> str:
    """Return the first documentation line for a serializer callable."""
    lines = str(getattr(fn, "__doc__", "") or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def _serializer_reference(serializer: SerializerFn, resolver: ModuleResolver) -> dict[str, Any]:
    """Return one durable executable reference without persisting executables."""
    from heavenbase.capsule.runtime import Capsule

    if not isinstance(serializer, Capsule):
        return python_executable_ref(serializer, label="MCP serializer")
    if serializer.manifest is None:
        raise ValueError("MCP serializer Capsule must be bound and explicitly registered")
    reference = capsule_executable_ref(serializer.manifest.id)
    try:
        resolver.resolve_executable(reference)
    except (ImportError, KeyError) as error:
        raise ValueError(f"MCP serializer Capsule {serializer.manifest.id!r} must be explicitly registered") from error
    return reference
