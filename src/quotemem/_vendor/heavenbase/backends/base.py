"""Physical backend runtime protocol."""

from __future__ import annotations

__all__ = ["Backend", "backend_locked"]

import inspect
import threading
from collections.abc import Mapping as AbcMapping, Sequence as AbcSequence, Set as AbcSet
from contextlib import AbstractContextManager, ExitStack, nullcontext
from functools import partial, wraps
from heapq import nsmallest
from operator import itemgetter
from weakref import WeakKeyDictionary

from heavenbase.utils import Any, Callable, Iterable, Iterator, Mapping, Sequence

from ..execution.frame.result import ResultFrame
from ..execution.contracts import QueryFragment, RowOp
from ..modules import ModuleResolver, ResolvedTagValue, TagView, module_resolver
from ..modules.tags import tag as module_tag, tags as module_tags
from ..schema.model import EntitySchema
from ..types import logical_type_from_any
from ..utils.ids import check_id, check_workspace_id
from heavenbase.utils.registry import class_identifier
from .capabilities import ScalarIndexSpec
from .mcp import BackendMCP
from .strategy import StrategyAdapter

# Only these forwarding wrappers preserve the signature for projection dispatch.
_SIGNATURE_WRAPPERS: WeakKeyDictionary[Callable[..., Any], Callable[..., Any]] = WeakKeyDictionary()


class _RegistrationPublication:
    """Commit SQL transactions changing existing shared indexes last."""

    def __init__(self) -> None:
        self._transactions: list[tuple[Any, ExitStack]] = []
        self._protected_connections: set[Any] = set()

    def stage(self, connection: Any, transaction: ExitStack) -> None:
        self._transactions.append((connection, transaction))

    def protect(self, connection: Any) -> None:
        self._protected_connections.add(connection)

    def must_commit_before_projecting(self, backends: Iterable[Backend]) -> bool:
        """Check whether a projection can escape a failed registration transaction."""
        if len(self._transactions) != 1:
            return bool(self._transactions)
        connection, _ = self._transactions[0]
        return any(getattr(getattr(backend, "db", None), "_conn", None) is not connection for backend in backends)

    def commit(self) -> None:
        for _, transaction in sorted(self._transactions, key=lambda item: item[0] in self._protected_connections):
            transaction.close()
        self._transactions.clear()


def backend_locked(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run a backend method under the instance re-entrant lock.

    Args:
        fn: Backend instance method to guard with ``self._lock``.

    Returns:
        Callable[..., Any]: Wrapped backend method.
    """

    @wraps(fn)
    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return fn(self, *args, **kwargs)

    _SIGNATURE_WRAPPERS[_wrapped] = fn
    return _wrapped


_KEYWORD_SUPPORT: dict[tuple[type, str, str], bool] = {}


def accepts_keyword(backend: Any, method: str, keyword: str) -> bool:
    """Return whether a backend method explicitly declares a keyword argument.

    A provider written against an earlier ``Backend`` may override a method with the
    signature it had then. A generic ``**kwargs`` can forward to an older
    method or ignore unknown options. The answer is
    cached per backend class; a method replaced on the instance itself is inspected
    on every call, since it can differ between instances of one class.

    Args:
        backend: Backend instance whose method is called.
        method: Method name, such as ``"get_many"``.
        keyword: Keyword argument name, such as ``"columns"``.

    Returns:
        bool: ``True`` when the signature names ``keyword`` as a keyword-capable parameter.
    """
    override = getattr(backend, "__dict__", {}).get(method)
    if override is not None:
        return _signature_accepts(override, keyword)
    key = (type(backend), method, keyword)
    supported = _KEYWORD_SUPPORT.get(key)
    if supported is None:
        supported = _KEYWORD_SUPPORT[key] = _signature_accepts(getattr(type(backend), method), keyword)
    return supported


def _signature_accepts(func: Callable[..., Any], keyword: str) -> bool:
    """Return whether the signature explicitly allows ``keyword=...``."""
    func = getattr(func, "__func__", func)
    while inspect.isfunction(func):
        wrapped = _SIGNATURE_WRAPPERS.get(func)
        if wrapped is None:
            break
        func = wrapped
    try:
        parameter = inspect.signature(func, follow_wrapped=False).parameters.get(keyword)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (parameter.KEYWORD_ONLY, parameter.POSITIONAL_OR_KEYWORD)


def read_projected(backend: Any, method: str, *args: Any, columns: Sequence[str] | None, **kwargs: Any) -> list[Any]:
    """Call a list-returning backend read with a column selection the provider may predate.

    ``columns`` is passed only when it is not ``None`` and the method names it.
    A provider whose method has no explicit ``columns`` parameter is called without it, and
    its complete fragments are trimmed here exactly as the portable ``Backend`` default
    trims them. Remove this fallback when pre-0.1.2.3 provider signatures stop being
    supported.

    Args:
        backend: Backend instance to read from.
        method: List-returning read method, such as ``"get_many"`` or ``"scan_page"``.
        *args: Positional arguments for the method.
        columns: Requested field names, or ``None`` for complete fragments.
        **kwargs: Other keyword arguments for the method.

    Returns:
        list[Any]: The method's row fragments projected to ``columns``, in its own shape
        (``get_many`` keeps ``None`` for missing rows; ``scan_page`` has none).

    Raises:
        TypeError: If ``columns`` is a single string or contains a non-string name.
    """
    read = getattr(backend, method)
    if columns is None:
        return list(read(*args, **kwargs))
    if accepts_keyword(backend, method, "columns"):
        return list(read(*args, columns=columns, **kwargs))
    return project_rows(read(*args, **kwargs), columns)


def project_rows(rows: Iterable[dict[str, Any] | None], columns: Sequence[str] | None) -> list[dict[str, Any] | None]:
    """Trim row fragments to a requested selection, preserving ``None`` for missing rows.

    Args:
        rows: Complete row fragments, with ``None`` for missing rows.
        columns: Requested field names, or ``None`` to keep fragments whole.

    Returns:
        list[dict[str, Any] | None]: Fragments holding ``object_id`` plus the requested fields they store.

    Raises:
        TypeError: If ``columns`` is a single string or contains a non-string name.
    """
    selection = Backend._row_columns(columns)
    return [None if row is None else Backend._project_row(row, selection) for row in rows]


# Enumeration layers, lowest first. The portable default of each upper layer reads the layer below it.
_LAYER_LEVELS = {"scan": 0, "iter_scan": 1, "scan_page": 2}
_ENUMERATION = threading.local()


def _override_frames() -> list[tuple[int, int]]:
    """Return this thread's stack of running enumeration overrides, as ``(id(backend), level)`` pairs."""
    frames = getattr(_ENUMERATION, "frames", None)
    if frames is None:
        frames = _ENUMERATION.frames = []
    return frames


def _below_override(backend: Any, level: int) -> bool:
    """Return whether an enumeration override of ``backend`` below ``level`` is running on this thread."""
    owner = id(backend)
    return any(frame_owner == owner and frame_level < level for frame_owner, frame_level in getattr(_ENUMERATION, "frames", ()))


def _portable_layer(name: str, native: Any) -> Callable[..., Any]:
    """Return the portable ``Backend`` layer ``name`` for a subclass that overrides a layer below its native one.

    The portable default reads the subclass's override. When the override
    calls this layer through ``self`` (a layer it does not override) on the
    thread running it, the call comes from inside a lower override and is
    answered by ``native``, the layer the subclass inherited; the override sits
    above it already, so reading through it again would recurse. Native code
    the override reaches through ``super()`` reads this layer through
    ``native_layer``, which answers with ``native`` on any thread.
    """
    portable = getattr(Backend, name)
    level = _LAYER_LEVELS[name]
    native_columns = _signature_accepts(native, "columns")

    # the subclass's override may hand reads of this backend to other threads, so no lock is held around it
    unlocked = Backend._portable_page if name == "scan_page" else portable

    @wraps(portable)
    def layer(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _below_override(self, level):
            return unlocked(self, *args, **kwargs)
        return answer_natively(self, *args, **kwargs)

    def answer_natively(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = native.__get__(self, type(self))
        columns = kwargs.get("columns")
        if columns is None or native_columns:
            return bound(*args, **kwargs)
        # a native layer written before ``columns`` answers complete fragments
        del kwargs["columns"]
        rows = bound(*args, **kwargs)
        if name == "scan_page":
            return project_rows(rows, columns)
        selection = Backend._row_columns(columns)
        return (Backend._project_row(dict(row), selection) for row in rows)

    setattr(layer, "_portable_of", portable)
    setattr(layer, "_native", native)
    setattr(layer, "_answer_natively", answer_natively)
    _SIGNATURE_WRAPPERS[layer] = portable
    return layer


def native_layer(backend: Any, name: str) -> Callable[..., Any]:
    """Return ``backend``'s enumeration layer ``name`` as the native code of a built-in provider must call it.

    A native layer that reads another layer of its own backend (the SQL
    ``iter_scan`` reads ``scan_page``) is reached on a subclass only through the
    subclass's override, so the layer it reads must stay native too: the
    portable fallback the subclass received in its place would read the override
    again. This holds whichever thread runs the native code.

    Args:
        backend: Backend instance whose layer is read.
        name: Layer name, ``"iter_scan"`` or ``"scan_page"``.

    Returns:
        Callable[..., Any]: The bound layer, or the native layer the subclass
        inherited when ``name`` is a portable fallback.
    """
    bound = getattr(backend, name)
    natively = getattr(getattr(bound, "__func__", None), "_answer_natively", None)
    return bound if natively is None else partial(natively, backend)


def _marked_override(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``fn`` recorded as a running enumeration override while it, or an iterator it returns, runs."""
    level = _LAYER_LEVELS[name]

    @wraps(fn)
    def override(self: Any, *args: Any, **kwargs: Any) -> Any:
        frames = _override_frames()
        frames.append((id(self), level))
        try:
            result = fn(self, *args, **kwargs)
        finally:
            frames.pop()
        return _marked_result((id(self), level), result)

    setattr(override, "_override_level", level)
    _SIGNATURE_WRAPPERS[override] = fn
    return override


def _marked_result(frame: tuple[int, int], result: Any) -> Any:
    """Record ``frame`` while a lazy result of an override runs; a materialized result is returned as is."""
    if isinstance(result, Iterator):
        return _marked_items(frame, result)
    if isinstance(result, Iterable) and not isinstance(result, (AbcSequence, AbcSet, AbcMapping)):
        # an iterable that is not an iterator (a cursor-like object) runs the override's code in ``__iter__``
        return _MarkedIterable(frame, result)
    return result


class _MarkedIterable:
    """An override's lazy iterable, recorded as the running override while it starts or produces an item."""

    def __init__(self, frame: tuple[int, int], items: Iterable[Any]) -> None:
        self._frame = frame
        self._items = items

    def __iter__(self) -> Iterator[Any]:
        frames = _override_frames()
        frames.append(self._frame)
        try:
            items = iter(self._items)
        finally:
            frames.pop()
        return _marked_items(self._frame, items)


def _marked_items(frame: tuple[int, int], items: Iterator[Any]) -> Iterator[Any]:
    """Yield ``items``, recording ``frame`` only while the override's own code produces the next one."""
    try:
        while True:
            frames = _override_frames()
            frames.append(frame)
            try:
                item = next(items)
            except StopIteration:
                return
            finally:
                frames.pop()
            yield item
    finally:
        close = getattr(items, "close", None)
        if callable(close):
            frames = _override_frames()
            frames.append(frame)
            try:
                close()
            finally:
                frames.pop()


def is_portable_layer(func: Any, name: str) -> bool:
    """Return whether ``func`` is ``Backend``'s portable layer ``name``, directly or as a subclass's fallback.

    Args:
        func: A backend method, bound or not.
        name: Layer name, ``"iter_scan"`` or ``"scan_page"``.

    Returns:
        bool: ``True`` for ``Backend.<name>`` and for the fallback a subclass that
        overrides a lower layer receives in its place.
    """
    func = getattr(func, "__func__", func)
    portable = getattr(Backend, name)
    return func is portable or getattr(func, "_portable_of", None) is portable


class Backend:
    """Thin physical adapter that executes fragments without parsing query specs.

    Provider adapters subclass ``Backend`` to implement physical row storage and
    compiled-fragment execution. Static behavior is declared on the published
    Backend-type record; construction and live values stay instance-local.

    Enumeration has three layers: ``scan`` (the exhaustive 0.1.2.2 contract),
    ``iter_scan`` (a bounded-memory pass) and ``scan_page`` (one keyset page).
    A subclass that overrides a lower layer below the class whose native upper
    layer it inherits (for example an ``InMemBackend`` or ``SQLiteBackend``
    subclass whose ``scan`` hides soft-deleted rows or hydrates rows lazily)
    gets the portable ``Backend.iter_scan`` and ``Backend.scan_page`` instead,
    which read through its override; the inherited native layer would read
    the parent's storage directly and bypass it. The override may itself call
    the parent's enumeration: ``super().scan`` and ``super().iter_scan`` read
    the inherited layers on any thread, and ``self.iter_scan`` or
    ``self.scan_page`` called from a ``scan`` override, on the thread running
    it, answers with the inherited native layer. A worker thread the override
    starts must use ``super()``: there ``self.iter_scan`` reaches the portable
    layer, which runs the override again. The portable layers take no backend
    lock of their own, so the override may hand reads of this backend to other
    threads. Such a subclass that still wants a native page defines its own
    ``iter_scan`` and ``scan_page``.

    Args:
        name: Workspace backend instance name.
        ws_id: Optional workspace namespace to bind at construction time.
    """

    identifier = "backend"
    aliases: Sequence[str] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Route enumeration through a subclass's own ``scan`` or ``iter_scan`` override.

        Each upper layer must be defined at least as far down the MRO as every
        layer it replaces; otherwise the subclass receives the portable default,
        which is written in terms of the lower layer. The portable default
        answers with the inherited native layer when it is reached from inside
        the subclass's lower override on the same thread, so an override that
        calls the parent's native enumeration does not recurse. To tell those
        calls apart, the ``scan`` and ``iter_scan`` of such a class (and of its
        subclasses) record that they are running, together with any iterator or
        lazy iterable they return while it produces items. Native code that
        reads another layer of its own backend reads it through
        ``native_layer``, which needs no such record.
        """
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__

        def depth(name: str) -> int:
            return next(index for index, klass in enumerate(mro) if name in vars(klass))

        def resolved(name: str) -> Any:
            return vars(mro[depth(name)])[name]

        def fall_back(name: str) -> None:
            # the inherited layer is native unless it is already Backend's portable one
            inherited = resolved(name)
            if inherited is not vars(Backend)[name]:
                setattr(cls, name, _portable_layer(name, getattr(inherited, "_native", inherited)))

        if depth("iter_scan") > depth("scan"):
            fall_back("iter_scan")
        if depth("scan_page") > min(depth("scan"), depth("iter_scan")):
            fall_back("scan_page")
        if any(hasattr(vars(klass).get(name), "_portable_of") for klass in mro for name in ("iter_scan", "scan_page")):
            for name in ("scan", "iter_scan"):
                method = resolved(name)
                marked = hasattr(method, "_portable_of") or hasattr(method, "_override_level")
                if method is not vars(Backend)[name] and inspect.isfunction(method) and not marked:
                    setattr(cls, name, _marked_override(name, method))

    def __init__(
        self,
        name: str,
        ws_id: str | None = None,
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> None:
        """Create a backend instance.

        Args:
            name: Workspace backend instance name.
            ws_id: Optional workspace namespace. Passing ``None`` leaves the backend unbound.
            resolver: Context-owned module authority. ``None`` retains an
                unresolved binding for direct bootstrap or standalone use.
            config: Config authority used by provider construction and runtime policy.

        Returns:
            None: This function does not return a value.
        """
        if config is not None and not callable(getattr(config, "get", None)):
            raise TypeError("Backend config authority must provide get(path, default)")
        self._module_resolver = resolver
        self._config = config
        self.name = check_id(name, "backend instance name")
        self.type = type(self).identifier
        self._ws_id: str | None = None
        self.ws_id = ws_id
        self._lock = threading.RLock()
        self._strategy_adapters: dict[str, StrategyAdapter] = {}

    @property
    def module_resolver(self) -> ModuleResolver:
        """Return the module authority bound to this live backend.

        Fully unbound standalone backends use the default Context resolver.
        A Backend with explicit config but no resolver is the private Context
        bootstrap state and cannot be consumed until Context finalizes it.
        Workspace registries bind both authorities before exposing a Backend.

        Returns:
            ModuleResolver: Bound or standalone-default module authority.

        Raises:
            RuntimeError: If a deferred Context bootstrap Backend escaped
                before its ModuleService was bound.
        """
        if self._module_resolver is None and self._config is not None:
            raise RuntimeError("Context bootstrap Backend is not yet bound to its ModuleService")
        return module_resolver(self._module_resolver)

    @property
    def config(self) -> Any:
        """Return the configuration authority bound to this live backend.

        Returns:
            Any: Bound ConfigManager-compatible object, or the process default
                Context's config for a standalone backend.
        """
        if self._config is not None:
            return self._config
        from heavenbase.context.authority import omitted_config

        return omitted_config()

    def _construction_config_get(self, path: str, default: Any = None) -> Any:
        """Read policy safely while direct control bootstrap has no Registry.

        Ordinary Backends consult their explicitly bound config authority.
        The direct control Backend must use that manager's immutable default
        layer until its Registry-backed scope store has opened, preventing the
        persistence Backend from recursively querying itself.
        """
        if self._module_resolver is None and self._config is not None:
            values = getattr(self._config, "default", None)
            if isinstance(values, Mapping):
                from ..utils.config.data import dget

                return dget(values, path, default)
        return self.config.get(path, default=default)

    @property
    def ws_id(self) -> str:
        """Return the bound workspace id.

        Returns:
            str: Bound workspace id, or ``""`` when this backend is unbound.
        """
        return self._ws_id or ""

    @ws_id.setter
    def ws_id(self, value: str | None) -> None:
        """Set the bound workspace id.

        Args:
            value: Workspace id to validate and store, or ``None`` to clear binding.

        Returns:
            None: This function does not return a value.
        """
        self._ws_id = check_workspace_id(value) if value else None

    @staticmethod
    def locked(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Run a backend method under the instance re-entrant lock.

        Args:
            fn: Backend instance method to wrap.

        Returns:
            Callable[..., Any]: Wrapped backend method.
        """
        return backend_locked(fn)

    @classmethod
    def from_config(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        resolver: ModuleResolver | None = None,
        config: Any | None = None,
    ) -> "Backend":
        """Build a backend instance from a config mapping.

        Args:
            name: Workspace backend instance name.
            data: Backend-specific config with optional ``ws_id``.
            resolver: Context-owned module authority.
            config: Config authority used by provider construction and runtime policy.

        Returns:
            Backend: Configured backend instance.
        """
        cfg = dict(data or {})
        return cls(name, ws_id=cfg.pop("ws_id", None), resolver=resolver, config=config, **cfg)

    @classmethod
    def register(
        cls,
        *,
        identifier: str | None = None,
        tags: Mapping[str, Any] | None = None,
        aliases: Sequence[str] | None = None,
        factory: Callable[[str, Mapping[str, Any]], "Backend"] | None = None,
        resolver: ModuleResolver | None = None,
    ) -> type["Backend"]:
        """Register this backend class and its config builder.

        Args:
            identifier: Optional canonical backend identifier. Defaults to the backend class's own
                ``identifier`` or a class-name-derived identifier.
            tags: Optional raw static Tag claims copied to the published Backend-type record.
            aliases: Optional identifier aliases. Defaults to aliases declared directly on ``cls``.
            factory: Optional importable `BackendFactory`. Defaults to
                adapting `cls.from_config` at resolution time.
            resolver: Context-owned module resolver. The default Context is
                selected when omitted. Backend registration requires the
                selected resolver to be a `ModuleService`.

        Returns:
            type[Backend]: The registered backend class.
        """
        from .registry import BackendType

        ident = identifier or class_identifier(cls, label="backend identifier")
        alias_values = tuple(cls.__dict__.get("aliases", ()) if aliases is None else aliases)
        return BackendType._register(
            cls,
            tags={} if tags is None else tags,
            aliases=alias_values,
            identifier=ident,
            factory=factory,
            resolver=module_resolver(resolver),
        )

    @classmethod
    def indexable(cls, dtype: Any, *, resolver: ModuleResolver | None = None) -> bool:
        """Return whether this backend's default value index can represent a logical type.

        Args:
            dtype: Logical type class, instance, identifier, alias, or descriptor.
            resolver: Context-owned module resolver. The default Context is
                selected when omitted.

        Returns:
            True when the backend's scalar equality/property index policy accepts the type.
        """
        selected = module_resolver(resolver)
        result = module_tag(selected, "backend_type", cls.identifier, "indexable-types")
        if result is None:
            return False
        if result.value is None or not isinstance(result.value, Sequence) or isinstance(result.value, (str, bytes)):
            raise TypeError(f"Backend {cls.identifier!r} indexable-types tag must be a sequence")
        return type(logical_type_from_any(dtype, resolver=selected)).identifier in result.value

    def _adopt_process_defaults(self) -> None:
        """Bind a fully unbound standalone Backend to process-default authorities.

        This is the only post-construction adoption contract. Explicit
        construction bindings are immutable, and custom Context registries
        must receive Backends built with their resolver and config directly.
        """
        if self._module_resolver is not None or self._config is not None:
            raise ValueError("Only a fully unbound standalone Backend can adopt process-default authorities")
        from heavenbase.context.authority import resolve_runtime_authority

        authority = resolve_runtime_authority(label="Backend")
        self._module_resolver = authority.resolver
        self._config = authority.config

    def _finalize_bootstrap_resolver(self, resolver: ModuleResolver) -> None:
        """Complete the direct control Backend's one deferred authority binding.

        Direct Context bootstrap constructs its persistence Backend before the
        module catalog can exist. The owning root Context calls this seam once
        after creating that catalog. Ordinary and workspace Backends must bind
        their resolver during construction instead.
        """
        if resolver is None:
            raise TypeError("Bootstrap resolver finalization requires a concrete module authority")
        if self._module_resolver is resolver:
            return
        if self._module_resolver is not None:
            raise ValueError("The control Backend is already bound to a different module resolver")
        if self._config is None:
            raise ValueError("Bootstrap resolver finalization requires an explicitly bound config authority")
        self._module_resolver = resolver

    @staticmethod
    def _physical_name(*parts: str, sep: str = "_") -> str:
        """Return a transparent provider name from already-validated id parts.

        This is intentionally a small backend helper, not a collision-resolution system. It only normalizes
        HeavenBase's public ``-`` / ``_`` separators to the provider separator and joins multiple parts. Users
        remain responsible for choosing logical ids that are unique, short enough, and portable for the
        backends they operate directly.
        """
        return sep.join(str(part).replace("-", sep).replace("_", sep) for part in parts)

    def schema_warnings(self, schema: EntitySchema, bindings: Sequence[Any]) -> list[str]:
        """Return optional backend-specific warnings for a schema before physical ensure.

        Backends override this when they can cheaply detect provider naming anti-patterns such as reserved
        identifiers, length limits, or normalized-name collisions. The default is intentionally silent.

        Args:
            schema: Entity schema being registered.
            bindings: Storage bindings routed to this backend.

        Returns:
            list[str]: User-facing warning messages.
        """
        return []

    def warn_schema(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Emit optional backend-specific schema warnings for user-facing schemas.

        Args:
            schema: Entity schema being registered.
            bindings: Storage bindings routed to this backend.

        Returns:
            None: This function does not return a value.
        """
        import warnings

        if "system" in schema.tags or self._is_internal_entity_id(schema.entity_id):
            return
        for message in self.schema_warnings(schema, bindings):
            warnings.warn(message, UserWarning, stacklevel=4)

    @staticmethod
    def _is_internal_entity_id(entity_id: str) -> bool:
        """Return whether an entity identifier is owned by HeavenBase internals."""
        return str(entity_id).startswith("sys-")

    def _physical_name_warnings(
        self,
        names: Sequence[tuple[str, str]],
        *,
        kind: str,
        max_length: int | None = None,
        reserved: Iterable[str] = (),
        case_sensitive: bool = True,
    ) -> list[str]:
        """Return warnings for provider-name length, reserved-word, and collision risks."""
        out: list[str] = []
        reserved_names = {str(name).lower() for name in reserved}
        pairs = [(str(logical), str(physical)) for logical, physical in names]
        for logical, physical in pairs:
            if max_length is not None and len(physical) > max_length:
                out.append(
                    f"Backend {self.name!r} ({self.type}) maps {kind} {logical!r} to provider name {physical!r} "
                    f"({len(physical)} chars), above limit {max_length}"
                )
            if physical.lower() in reserved_names:
                out.append(f"Backend {self.name!r} ({self.type}) maps {kind} {logical!r} to reserved provider name {physical!r}")
        seen: dict[str, list[tuple[str, str]]] = {}
        for logical, physical in pairs:
            key = physical if case_sensitive else physical.lower()
            seen.setdefault(key, []).append((logical, physical))
        for matches in seen.values():
            if len(matches) > 1:
                details = ", ".join(f"{logical!r}->{physical!r}" for logical, physical in matches)
                out.append(f"Backend {self.name!r} ({self.type}) maps multiple {kind} names to the same provider name: {details}")
        return out

    def bind_workspace(self, ws_id: str) -> "Backend":
        """Bind this backend instance to a validated workspace namespace.

        Args:
            ws_id: Workspace namespace to bind.

        Returns:
            Backend: This backend instance for chained setup.
        """
        with self._lock:
            next_id = check_workspace_id(ws_id)
            if self._ws_id != next_id:
                self._ws_id = next_id
                self._refresh_workspace()
        return self

    def _refresh_workspace(self) -> None:
        """Refresh state derived from the workspace namespace."""
        return None

    def drop(self) -> None:
        """Drop this backend's current workspace namespace, if it has one.

        Args:
            None.

        Returns:
            None: This function does not return a value.
        """
        return None

    def ensure(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Prepare storage for an entity schema and its bindings.

        Args:
            schema: Entity schema to create or refresh in physical storage.
            bindings: Storage bindings routed to this backend.

        Returns:
            None: This function does not return a value.
        """
        return None

    def ensure_plan(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Prepare storage from a resolved plan while preserving the legacy ensure seam.

        Backends that implement configurable scalar indexes override this method.
        Other providers continue receiving the historical storage-binding sequence.

        Args:
            schema: Entity schema to create or refresh in physical storage.
            storage_plan: Resolved placement, index, and constraint plan.

        Returns:
            None: This function does not return a value.
        """
        self.ensure(schema, storage_plan.bindings)
        self.ensure_indexes(schema, storage_plan)

    def _stage_registration(
        self,
        schema: EntitySchema,
        storage_plan: Any,
        *,
        peers: Sequence[Backend] = (),
        publication: _RegistrationPublication | None = None,
        reattaching: bool = False,
    ) -> AbstractContextManager[None]:
        """Prepare storage and return the context for strategy changes during publication."""
        self.ensure_plan(schema, storage_plan)
        return nullcontext()

    def _decide_registration_aliases(self, schema: EntitySchema, storage_plan: Any, peers: Sequence[Backend]) -> None:
        """Decide which routes share this backend's storage, after every route is staged and before publication."""

    def _prepare_registration_scan(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Open a stored entity for reading without reconciling scalar indexes."""
        self.ensure(schema, storage_plan.bindings)

    def _validate_registration_reattachment(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Check whether the proposed old-mode bindings can use stored strategy state."""

    def _stored_scalar_index_exists(self, entity_id: str, fields: Sequence[str], *, unique: bool) -> bool | None:
        """Report whether a physical scalar index exists, or whether its presence is unknown."""
        return None

    def _prune_registration(self, schema: EntitySchema, bindings: Sequence[Any], peers: Sequence[Backend], *, reattaching: bool) -> None:
        """Release unused strategy state after registration."""
        if not reattaching:
            self.prune(schema, bindings)

    def _row_write_routes(self, entity_id: str) -> tuple[str, ...]:
        """Return the workspace routes that write one physical row together."""
        return (self.name,)

    def _write_workspace_rows(self, rows: Sequence[RowOp]) -> None:
        """Write workspace mutations after routing fragments to physical rows."""
        self.upsert(rows)

    @backend_locked
    def prune(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Release strategy state for fields of an entity that a registration no longer binds here.

        ``ensure`` adds strategy state and never releases it (``ensure_plan``
        also reconciles scalar indexes, which drops the ones the new plan no
        longer wants), so a strategy index the previous registration uses stays
        in place and maintained while a new registration is being ensured. The
        workspace calls this once the registration that uses ``bindings`` is
        published (and ``abandon`` when a registration fails before then), so a
        strategy index (a sparse-GRAM keyword index, say) is released only when
        the live registration no longer binds it. Through ``StrategyAdapter.prune``,
        every strategy adapter receives the bindings routed to this backend that it
        supports, and releases the state of the fields it indexes outside them.

        Args:
            schema: Entity schema of the registration that ``bindings`` belong to.
            bindings: The entity's storage bindings under that registration.

        Returns:
            None: This function does not return a value.
        """
        mine = [binding for binding in bindings if binding.backend == self.name]
        for adapter in self._strategy_adapters.values():
            adapter.prune(schema, [binding for binding in mine if adapter.supports(schema, binding)])

    @backend_locked
    def abandon(self, schema: EntitySchema, bindings: Sequence[Any]) -> None:
        """Undo the strategy state that a registration which failed before it was published set up here.

        The workspace calls this on every backend the failed registration
        ensured, with the bindings of the registration that stays live (none
        when the entity had no registration). Unlike ``prune``, it never
        releases stored strategy state that existed before the failed
        registration began: another workspace on the same database may still use
        it, for example the shared sparse-GRAM index on a backend that the
        entity's live registration does not use. Every strategy adapter
        receives, through ``StrategyAdapter.abandon``, the bindings routed to
        this backend that it supports.

        Args:
            schema: Entity schema of the registration that stays live, or of the failed one when none does.
            bindings: The entity's storage bindings under the registration that stays live.

        Returns:
            None: This function does not return a value.
        """
        mine = [binding for binding in bindings if binding.backend == self.name]
        for adapter in self._strategy_adapters.values():
            adapter.abandon(schema, [binding for binding in mine if adapter.supports(schema, binding)])

    def ensure_indexes(self, schema: EntitySchema, storage_plan: Any) -> None:
        """Reconcile configurable scalar indexes after structural ensure.

        Args:
            schema: Entity schema whose resolved index plan is active.
            storage_plan: Resolved placement, index, and constraint plan.

        Returns:
            None: Providers without configurable scalar indexes do nothing.
        """
        return None

    @staticmethod
    def ordered_row_op_batches(rows: Sequence[RowOp]) -> Iterable[tuple[str, bool, list[RowOp]]]:
        """Yield efficient mutation batches without changing observable input order.

        A batch contains adjacent operations for one entity and one operation kind.
        Repeating an object id starts a new batch so adapters that implement replacement
        with ``delete + insert`` still preserve sequential same-key semantics.

        Args:
            rows (Sequence[RowOp]): Ordered backend mutations.

        Yields:
            tuple[str, bool, list[RowOp]]: Entity id, delete flag, and a non-empty
                same-kind batch with unique object ids.
        """
        entity_id: str | None = None
        is_delete = False
        batch: list[RowOp] = []
        object_ids: set[str] = set()
        for row in rows:
            row_is_delete = row.is_delete()
            object_id = str(row.object_id)
            if batch and (row.entity_id != entity_id or row_is_delete != is_delete or object_id in object_ids):
                assert entity_id is not None
                yield entity_id, is_delete, batch
                batch = []
                object_ids = set()
            if not batch:
                entity_id = row.entity_id
                is_delete = row_is_delete
            batch.append(row)
            object_ids.add(object_id)
        if batch:
            assert entity_id is not None
            yield entity_id, is_delete, batch

    def validate_missing_row(self, entity_id: str, object_id: str, *, for_patch: bool = False) -> None:
        """Check whether a mutation can proceed after this backend returned no row.

        The default accepts absence. A provider whose reads can miss stored
        rows may reject the mutation before any backend is written. With
        ``for_patch=False`` the workspace asks whether the miss shows that the
        row is absent: it has read every route of the entity, none returned the
        row, and one route that accepts suffices, so accepting asserts that a
        miss here shows the row is absent. That holds when this backend keeps
        every row written to it, since the workspace writes every row to every
        route its entity has at the time. A replacement registration that adds
        a route does not copy the rows stored before, so the workspace asks
        only the entity's oldest routes, those it has had at least as long as
        any other.

        Args:
            entity_id: The entity routed to this backend.
            object_id: The object whose read returned no fragment.
            for_patch: Whether the mutation needs this route's values for a patch
                of a row that exists, rather than the row's absence.

        Raises:
            ValueError: If the provider cannot safely proceed after a missed read.
        """

    def settle_query_misses(self, entity_id: str, object_ids: Iterable[Any], *, _near_fragment: Any = None) -> dict[str, dict[str, Any]]:
        """Read again query rows whose hydration missed them and whose absence ``validate_missing_row`` left unsettled.

        The workspace asks this only for rows that a query returned, that no route
        returned when the workspace read them by id, and whose absence no route
        accepts. A row it returns is hydrated; any other such row keeps the values
        the query returned. The default reads nothing: a backend that keeps every
        row it is given accepts its misses in ``validate_missing_row``.

        Args:
            entity_id: The entity routed to this backend.
            object_ids: Object ids that the hydration read missed.
            _near_fragment: The near fragment whose provider route the query used, if any.

        Returns:
            dict[str, dict[str, Any]]: Rows that a later read returned, by object id.
        """
        return {}

    def validate_row_ops(self, rows: Sequence[RowOp]) -> None:
        """Check provider-specific row constraints before a workspace writes any route.

        The default accepts all operations. Overrides may inspect local backend
        state and must leave rows and storage untouched. A backend still applies
        its own constraints in ``upsert`` for direct callers.

        Args:
            rows: The planned mutations for this backend.

        Raises:
            ValueError: If any mutation fails a provider-specific constraint.
        """

    def upsert(self, rows: Sequence[RowOp]) -> None:
        """Apply replacement upserts and idempotent deletes for backend row fragments.

        Args:
            rows: Row mutations for this backend. Each ``RowOp`` has an ``entity_id``,
                ``object_id``, backend-ready encoded ``values``, and an ``op`` of either
                ``"upsert"`` or ``"delete"``. Use ``row.is_upsert()``,
                ``row.is_delete()``, and ``row.to_row()`` for dispatch and replacement
                row construction.

        Returns:
            None: This function does not return a value.

        Raises:
            NotImplementedError: If called on the base backend contract instead of a provider
                implementation.

        Notes:
            ``"upsert"`` replaces the stored fragment for ``(entity_id, object_id)`` with
            ``row.to_row()``. ``"delete"`` removes that fragment if present and succeeds when
            it is already absent. Callers may order mutations to preserve referential safety
            across a non-transactional apply. A provider must preserve that observable order;
            it may reorder only inside a batch whose failure is atomic.
        """
        raise NotImplementedError

    def upsert_owned(self, rows: Sequence[RowOp], *, owner_field: str, owner: str) -> list[RowOp]:
        """Apply mutations conditional on ownership in physical storage.

        Args:
            rows: Backend-ready mutations with distinct keys.
            owner_field: Stored field used to check current ownership.
            owner: Owner value required for a replacement or deletion.

        Returns:
            Applied mutations. An absent key accepts a create but needs no delete.

        Raises:
            NotImplementedError: If the backend cannot guard ownership at mutation time.

        Notes:
            Skip occupied keys with a different owner, including a missing owner
            value. A competing create may surface as a write conflict. Maintain
            strategy state for applied operations only.
        """
        raise NotImplementedError(f"Backend {self.type!r} does not support ownership-guarded mutations")

    def _upsert_owned_under_lock(self, rows: Sequence[RowOp], *, owner_field: str, owner: str) -> list[RowOp]:
        """Apply guarded mutations when the caller holds the lock and storage excludes other workspace writers."""
        applied = []
        for entity_id, is_delete, batch in self.ordered_row_op_batches(rows):
            existing = self.get_many(entity_id, [row.object_id for row in batch])
            for row, current in zip(batch, existing):
                if (current is None and not is_delete) or (current is not None and current.get(owner_field) == owner):
                    applied.append(row)
        if applied:
            self.upsert(applied)
        return applied

    def compare_upsert(
        self,
        row: RowOp,
        *,
        revision_field: str,
        expected_revision: int | None,
    ) -> bool:
        """Conditionally create or replace one row by revision.

        Providers that qualify for shared persistent Registry writes override
        this method with one atomic native operation. There is deliberately no
        read-then-upsert fallback because it would lose concurrent writes.

        Args:
            row (RowOp): Replacement upsert containing the next revision.
            revision_field (str): Row field holding the current integer revision.
            expected_revision (int | None): Required current revision. `None`
                requires the row to be absent.

        Returns:
            bool: `True` when the replacement won; `False` when the expected
            state did not match and no mutation occurred.

        Raises:
            NotImplementedError: If this Backend has not qualified atomic CAS.
        """
        raise NotImplementedError(f"Backend {self.type!r} does not support atomic compare_upsert")

    def _guard_registration_rows(self, entity_id: str, object_ids: Sequence[str]) -> AbstractContextManager[Any]:
        """Hold off concurrent writers during MetaSchema compensation."""
        raise NotImplementedError(f"Backend {self.type!r} cannot conditionally restore MetaSchema rows")

    def delete(self, entity_id: str, object_ids: Iterable[Any]) -> None:
        """Delete object ids for one entity through the ``upsert`` mutation funnel.

        Args:
            entity_id: Logical entity identifier.
            object_ids: Object ids to remove.

        Returns:
            None: This function does not return a value.
        """
        self.upsert([RowOp(entity_id, object_id, op="delete") for object_id in object_ids])

    def get_many(self, entity_id: str, object_ids: Iterable[Any], *, columns: Sequence[str] | None = None) -> list[dict[str, Any] | None]:
        """Return rows for object ids in input order, using scan as the portable fallback.

        Args:
            entity_id: Logical entity identifier.
            object_ids: Object ids to read.
            columns: Optional field names to return. ``None`` returns the complete fragment;
                a selection returns ``object_id`` plus the named fields that this backend
                stores. Providers with a physical projection fetch only those fields;
                the portable default trims the complete fragment in Python.

        Returns:
            list[dict[str, Any] | None]: Row fragments aligned to the input ids. Missing rows are ``None``.

        Raises:
            TypeError: If ``columns`` is a single string or contains a non-string name.
        """
        ordered_ids = list(object_ids)
        wanted = set(ordered_ids)
        rows = {row.get("object_id"): row for row in self.scan(entity_id) if row.get("object_id") in wanted}
        return self._project_rows([rows.get(row_id) for row_id in ordered_ids], columns)

    @staticmethod
    def _row_columns(columns: Sequence[str] | None) -> tuple[str, ...] | None:
        """Normalize a requested column selection to a sorted tuple that always names ``object_id``.

        Args:
            columns: Requested field names, or ``None`` for the complete fragment.

        Returns:
            tuple[str, ...] | None: Sorted unique names including ``object_id``, or ``None``.

        Raises:
            TypeError: If ``columns`` is a single string instead of a sequence of
                names, or any requested name is not a string.
        """
        if columns is None:
            return None
        if isinstance(columns, str):
            raise TypeError("columns must be a sequence of field names, not a single str")
        if isinstance(columns, bytes):
            raise TypeError("columns must be a sequence of str field names, not bytes")
        names = list(columns)
        if not all(isinstance(name, str) for name in names):
            raise TypeError("columns must be field names")
        return tuple(sorted({"object_id", *names}))

    @staticmethod
    def _project_row(row: dict[str, Any], selection: tuple[str, ...] | None) -> dict[str, Any]:
        """Return the fragment fields named by a normalized selection, or the fragment itself for ``None``."""
        if selection is None:
            return row
        return {name: value for name, value in row.items() if name in selection}

    def _project_rows(self, rows: Iterable[dict[str, Any] | None], columns: Sequence[str] | None) -> list[dict[str, Any] | None]:
        """Trim lookup results to a requested selection, preserving ``None`` for missing rows."""
        return project_rows(rows, columns)

    def exists_many(self, entity_id: str, object_ids: Iterable[Any]) -> list[bool]:
        """Return existence flags for object ids in input order.

        Args:
            entity_id: Logical entity identifier.
            object_ids: Object ids to test.

        Returns:
            list[bool]: Boolean flags aligned to the input ids.
        """
        return [row is not None for row in self.get_many(entity_id, object_ids)]

    def count(self, entity_id: str) -> int:
        """Return row count for one entity.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            int: Number of row fragments stored for the entity.
        """
        return len(self.scan(entity_id))

    def exact_count(self, entity_id: str) -> int:
        """Return an exact row-fragment count suitable for integrity proofs.

        The default derives the proof from the exhaustive :meth:`scan`
        contract. A lookup-only backend must explicitly override this method;
        an ordinary provider count is never assumed exact by default.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            int: Exact number of row fragments stored for the entity.

        Raises:
            NotImplementedError: If this backend cannot prove an exact count.
        """
        return len(self.scan(entity_id))

    def execute(self, fragment: QueryFragment) -> ResultFrame:
        """Execute a compiled query fragment with the portable scan fallback.

        Args:
            fragment: Backend-ready query fragment.

        Returns:
            ResultFrame: Rows matched by the fragment.

        Raises:
            NotImplementedError: If the backend does not implement ``scan``.
        """
        from .scanning import execute_scan

        return execute_scan(self, fragment)

    def scan(self, entity_id: str) -> list[dict[str, Any]]:
        """Return all rows for one entity as dictionaries.

        This is an exhaustive integrity contract, not a best-effort page. Providers
        must paginate to exhaustion or raise when completeness cannot be guaranteed.

        Args:
            entity_id: Logical entity identifier.

        Returns:
            list[dict[str, Any]]: Backend row fragments.

        Raises:
            NotImplementedError: If the backend does not implement scanning.
        """
        raise NotImplementedError

    def iter_scan(self, entity_id: str, *, batch_size: int, columns: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
        """Iterate all unique row fragments, with bounded transient memory where the provider implements it.

        Args:
            entity_id (str): Logical entity identifier.
            batch_size (int): Positive maximum provider page size.
            columns (Sequence[str] | None): Optional field names to return, as for ``get_many``.
                ``None`` yields complete fragments.

        Returns:
            Iterable[dict[str, Any]]: Exhaustive detached row fragments.

        Raises:
            ValueError: If ``batch_size`` is not a positive integer.
            NotImplementedError: If the provider implements neither this method
                nor ``scan``.

        Notes:
            This default yields the rows of one ``scan`` call, so a provider
            written before ``iter_scan`` existed (every 0.1.2.2 backend) keeps
            working with ``iter_rows``, exact ``near`` and Catalog reconciliation,
            but holds the whole entity in memory once per enumeration; a provider
            with a native cursor overrides this to bound its memory. No backend
            lock is held between items of this default. This is not a
            cross-backend transaction or immutable snapshot. Callers must keep
            the source stable during exhaustive enumeration. A generator that
            holds the backend lock across ``yield`` must only be consumed while
            no other lock is taken between items; page-wise consumers use
            ``scan_page`` instead.
        """
        self._check_batch_size(batch_size)
        if getattr(self.scan, "__func__", None) is Backend.scan:
            raise NotImplementedError(f"{type(self).__name__} supports neither iter_scan nor scan")
        selection = self._row_columns(columns)
        return self._materialized_scan(entity_id, selection)

    def _materialized_scan(self, entity_id: str, selection: tuple[str, ...] | None) -> Iterable[dict[str, Any]]:
        """Yield the detached, projected rows of one ``scan`` call, which runs at the first item."""
        for row in self.scan(entity_id):
            yield self._project_row(dict(row), selection)

    def scan_page(self, entity_id: str, *, after: Any, batch_size: int, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return one page of row fragments in ``object_id`` order, strictly after ``after``.

        The page is returned as a detached list and no backend lock is held once
        it returns, so a caller may release its own locks and write between
        pages. Whether the lock is held while the page is fetched is the
        provider's choice: the SQL backend plans the page under the lock and
        runs the query outside it. This portable default holds the lock while it
        drives a provider's own ``iter_scan``, so an enumeration that pages by
        offset sees no write of this backend. Over the default ``iter_scan``,
        and on a subclass that receives this default because it overrides
        ``scan`` or ``iter_scan`` below an inherited native page, it takes no
        lock of its own and calls ``scan`` as 0.1.2.2 called it, so such a
        ``scan`` or override may hand reads of this backend to other threads.
        Pages are therefore not a snapshot: a row written behind the cursor is
        not seen, one written ahead of it is.

        Args:
            entity_id (str): Logical entity identifier.
            after (Any): ``object_id`` the page must follow, or ``None`` for the first page.
            batch_size (int): Positive maximum number of rows in the page.
            columns (Sequence[str] | None): Optional field names to return, as for ``get_many``.

        Returns:
            list[dict[str, Any]]: Detached row fragments in ascending ``object_id``
            order, at most ``batch_size`` of them; an empty list ends the scan.

        Raises:
            ValueError: If ``batch_size`` is not a positive integer.
            NotImplementedError: If the provider implements neither ``iter_scan`` nor ``scan``.

        Notes:
            This portable default drives ``iter_scan`` to exhaustion, closes
            the generator, and keeps the ``batch_size`` smallest ids beyond
            ``after``, so every page costs one complete
            enumeration and a scan of N rows costs O(N²/batch_size); over the
            default ``iter_scan`` that is one ``scan`` call per page. It is
            correct for providers whose bounded scan has no order; a provider
            with an ordered keyset scan overrides it with one page query, as the
            SQL and in-memory backends do. A subclass that overrides ``scan``
            or ``iter_scan`` below such a native page pages through this
            default instead, so its override is honoured.
        """
        if is_portable_layer(getattr(self, "iter_scan", None), "iter_scan"):
            return self._portable_page(entity_id, after=after, batch_size=batch_size, columns=columns)
        with self._lock:
            return self._portable_page(entity_id, after=after, batch_size=batch_size, columns=columns)

    def _portable_page(self, entity_id: str, *, after: Any, batch_size: int, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return the ``batch_size`` smallest ids beyond ``after`` from one closed ``iter_scan`` pass, taking no lock."""
        self._check_batch_size(batch_size)
        # A provider whose iter_scan predates ``columns`` answers complete fragments; the page is trimmed below.
        pushed = columns is not None and accepts_keyword(self, "iter_scan", "columns")
        selection = {"columns": columns} if pushed else {}
        rows = iter(self.iter_scan(entity_id, batch_size=batch_size, **selection))
        try:
            beyond = rows if after is None else (row for row in rows if row["object_id"] > after)
            page = nsmallest(batch_size, beyond, key=itemgetter("object_id"))
        finally:
            close = getattr(rows, "close", None)
            if callable(close):
                close()
        return page if pushed or columns is None else project_rows(page, columns)

    def end_scan(self, entity_id: str, *, after: Any) -> None:
        """End a ``scan_page`` enumeration that stops before its empty last page.

        A provider that keeps state for the page after ``after`` drops it here;
        ``ws.iter_rows`` calls this when its iterator is closed early. Without
        the call such state is only bounded by the provider. This default keeps
        no state and does nothing.

        Args:
            entity_id (str): Logical entity identifier of the enumeration.
            after (Any): Last ``object_id`` of the enumeration's last page.

        Returns:
            None: This function does not return a value.
        """
        del entity_id, after

    @staticmethod
    def _check_batch_size(batch_size: Any) -> None:
        """Reject a page size that is not a positive integer.

        Raises:
            ValueError: If ``batch_size`` is a boolean, not an integer, or not positive.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Return backend metadata suitable for system catalog rows.

        Args:
            None.

        Returns:
            dict[str, Any]: Backend instance metadata.
        """
        return {"name": self.name, "type": self.type}

    def _construction_tag_claims(self) -> dict[str, Any]:
        """Return construction-derived tag values for this concrete instance.

        Args:
            None.

        Returns:
            dict[str, Any]: Whole-value construction overlays.
        """
        return {}

    def _live_tag_claims(self) -> dict[str, Any]:
        """Return best-effort live tag values for this concrete instance.

        Args:
            None.

        Returns:
            dict[str, Any]: Whole-value live overlays.
        """
        return {"strategy-adapters": [adapter.to_dict() for adapter in self._strategy_adapters.values()]}

    def tags(self) -> TagView:
        """Return this concrete Backend's effective snapshot-bound Tag view.

        Args:
            None.

        Returns:
            TagView: Static record claims overlaid by construction and live values.
        """
        view = module_tags(self.module_resolver, "backend_type", self.type)
        construction = self._construction_tag_claims()
        if construction:
            view = view.overlay(construction, source="construction", instance=self)
        live = self._live_tag_claims()
        if live:
            view = view.overlay(live, source="live", instance=self)
        return view

    def tag(self, identifier: str) -> ResolvedTagValue | None:
        """Return one effective Tag result for this concrete Backend.

        Args:
            identifier: Canonical Tag identifier, alias, or unknown raw key.

        Returns:
            ResolvedTagValue | None: Snapshot-bound effective value, or `None`
                when neither a claim nor an applicable definition default exists.
        """
        return self.tags().get(identifier)

    def graph_incoming_sources(
        self,
        entity_id: str,
        field_name: str,
        target_entity: str,
        target_ids: Sequence[Any],
    ) -> list[str]:
        """Return a complete superset of source ids referencing typed targets.

        The workspace hydrates and rechecks returned sources before mutation, so false
        positives are safe but false negatives violate the capability contract.

        Args:
            entity_id (str): Logical entity that owns the GraphEdge field.
            field_name (str): GraphEdge field whose incoming references are requested.
            target_entity (str): Logical entity type of every requested target.
            target_ids (Sequence[Any]): Target object ids whose incoming sources are needed.

        Returns:
            list[str]: Source object ids containing every matching incoming reference;
                additional false-positive candidates are permitted.

        Raises:
            NotImplementedError: If native incoming lookup is not supported.
        """
        del entity_id, field_name, target_entity, target_ids
        raise NotImplementedError(f"{type(self).__name__} does not support native GraphEdge incoming lookup")

    def graph_traverse_refs(
        self,
        entity_id: str,
        start_id: Any,
        field_name: str,
        *,
        min_hops: int = 1,
        max_hops: int = 1,
        direction: str = "out",
        endpoint_preds: Sequence[tuple] | None = None,
    ) -> list[tuple[str, str]]:
        """Run native GraphEdge traversal and return typed node refs.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object id.
            field_name: GraphEdge field name.
            min_hops: Minimum traversal depth.
            max_hops: Maximum traversal depth.
            direction: Traversal direction. Supported values are backend-defined but normally ``"out"``, ``"in"``, or ``"both"``.
            endpoint_preds: Optional endpoint predicates already proven pushable by the route planner.

        Returns:
            list[tuple[str, str]]: ``(entity_id, object_id)`` refs reached by traversal.

        Raises:
            NotImplementedError: If the backend does not support native GraphEdge traversal.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support native GraphEdge traversal")

    def graph_traverse(
        self,
        entity_id: str,
        start_id: Any,
        field_name: str,
        *,
        min_hops: int = 1,
        max_hops: int = 1,
        direction: str = "out",
        endpoint_preds: Sequence[tuple] | None = None,
    ) -> list[str]:
        """Run native GraphEdge traversal and return object ids for self-referential callers.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object id.
            field_name: GraphEdge field name.
            min_hops: Minimum traversal depth.
            max_hops: Maximum traversal depth.
            direction: Traversal direction.
            endpoint_preds: Optional endpoint predicates already proven pushable by the route planner.

        Returns:
            list[str]: Reached object ids when the traversal stays inside one entity.
        """
        return [
            object_id
            for _entity, object_id in self.graph_traverse_refs(
                entity_id,
                start_id,
                field_name,
                min_hops=min_hops,
                max_hops=max_hops,
                direction=direction,
                endpoint_preds=endpoint_preds,
            )
        ]

    def graph_shortest_path_refs(
        self,
        entity_id: str,
        start_id: Any,
        end_id: Any,
        field_name: str,
        *,
        max_hops: int = 10,
        direction: str = "out",
    ) -> list[tuple[str, str]] | None:
        """Run native GraphEdge shortest-path search and return typed node refs.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object id.
            end_id: Target object id.
            field_name: GraphEdge field name.
            max_hops: Maximum path length to search.
            direction: Traversal direction.

        Returns:
            list[tuple[str, str]] | None: Path refs, or ``None`` when no path exists.

        Raises:
            NotImplementedError: If the backend does not support native GraphEdge traversal.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support native GraphEdge traversal")

    def graph_shortest_path(
        self,
        entity_id: str,
        start_id: Any,
        end_id: Any,
        field_name: str,
        *,
        max_hops: int = 10,
        direction: str = "out",
    ) -> list[str] | None:
        """Run native GraphEdge shortest-path search and return object ids for self-referential callers.

        Args:
            entity_id: Source entity identifier.
            start_id: Source object id.
            end_id: Target object id.
            field_name: GraphEdge field name.
            max_hops: Maximum path length to search.
            direction: Traversal direction.

        Returns:
            list[str] | None: Path object ids, or ``None`` when no path exists.
        """
        refs = self.graph_shortest_path_refs(entity_id, start_id, end_id, field_name, max_hops=max_hops, direction=direction)
        return [object_id for _entity, object_id in refs] if refs is not None else None

    def register_strategy_adapter(self, adapter: StrategyAdapter, *, replace: bool = False) -> StrategyAdapter:
        """Register a backend-local adapter for one storage strategy.

        Args:
            adapter (StrategyAdapter): Backend-local strategy adapter owned by this
                instance.
            replace (bool): Replace a different adapter for the same strategy.

        Returns:
            StrategyAdapter: The registered adapter.

        Raises:
            TypeError: If `adapter` is not a complete `StrategyAdapter` implementation.
            ValueError: If the adapter belongs to another backend or its strategy is
                already owned, or its identifier conflicts with another strategy.
        """
        if not isinstance(adapter, StrategyAdapter):
            raise TypeError(f"Expected StrategyAdapter, got {type(adapter).__name__}")
        if adapter.backend is not self:
            raise ValueError(f"Strategy adapter {adapter.identifier!r} belongs to a different backend")
        missing = [name for name in ("ensure", "upsert") if getattr(type(adapter), name) is getattr(StrategyAdapter, name)]
        if missing:
            raise TypeError(f"Strategy adapter {adapter.identifier!r} must implement {', '.join(missing)}")
        for registered in self._strategy_adapters.values():
            if registered.identifier == adapter.identifier and registered.strategy != adapter.strategy:
                raise ValueError(f"Backend {self.name!r} already has strategy adapter identifier {adapter.identifier!r}")
        existing = self._strategy_adapters.get(adapter.strategy)
        if existing is adapter:
            return adapter
        if existing is not None and not replace:
            raise ValueError(f"Backend {self.name!r} already has an adapter for strategy {adapter.strategy!r}")
        if existing is not None:
            existing.reset()
        self._strategy_adapters[adapter.strategy] = adapter
        return adapter

    def strategy_adapter(self, strategy: Any) -> StrategyAdapter | None:
        """Return the adapter registered for a strategy, if any.

        Args:
            strategy (Any): Storage strategy class, instance, identifier, alias,
                or serialized spec.

        Returns:
            StrategyAdapter | None: Registered adapter, or ``None`` when no adapter owns the strategy.
        """
        from ..strategies import StorageStrategy

        return self._strategy_adapters.get(StorageStrategy.identifier_of(strategy, resolver=self.module_resolver))

    def strategy_adapter_for_fragment(self, fragment: QueryFragment) -> StrategyAdapter | None:
        """Return the adapter that owns a compiled fragment, if any.

        Args:
            fragment (QueryFragment): Backend-ready query fragment.

        Returns:
            StrategyAdapter | None: Adapter that can execute the fragment, or ``None``.
        """
        for adapter in self._strategy_adapters.values():
            if adapter.can_execute(fragment):
                return adapter
        return None

    def supports_strategy(self, schema: EntitySchema, binding: Any) -> bool:
        """Return whether this backend has an adapter for one storage binding.

        Args:
            schema (EntitySchema): Entity schema being registered or queried.
            binding (Any): Storage binding to test.

        Returns:
            bool: True when a registered strategy adapter supports the binding.
        """
        adapter = self.strategy_adapter(getattr(binding, "strategy", None))
        return bool(adapter and adapter.supports(schema, binding))

    def candidate_id_fragment(self, object_ids: Sequence[Any]) -> QueryFragment:
        """Return a backend-local object-id filter fragment for candidate prefiltering.

        Args:
            object_ids: Candidate object ids to retain.

        Returns:
            QueryFragment: Python-scan fragment that filters rows by ``object_id``.
        """
        id_set = set(object_ids)

        def _payload(row: dict) -> bool:
            return row.get("object_id") in id_set

        return QueryFragment(self.name, "in", "object_id", _payload, notes={"mode": "py_scan", "row_payload": _payload, "candidate_ids": list(object_ids)})

    def scalar_index(self) -> ScalarIndexSpec:
        """Return the effective ``scalar-index`` Tag as a typed Backend facade.

        Args:
            None.

        Returns:
            ScalarIndexSpec: Effective scalar-index policy for this backend instance.
        """
        result = self.tag("scalar-index")
        return ScalarIndexSpec() if result is None else ScalarIndexSpec.load(result.value)

    def defaults_scalar_index(self, dtype: Any) -> bool:
        """Return whether this backend indexes one scalar type by default.

        Args:
            dtype: Logical type class, instance, identifier, alias, or descriptor.

        Returns:
            bool: Whether an undeclared field receives a provider-default index.
        """
        identifier = type(logical_type_from_any(dtype, resolver=self.module_resolver)).identifier
        caps = self.scalar_index()
        return caps.value and identifier in caps.default_types and self.indexable(dtype, resolver=self.module_resolver)

    def supports_scalar_index(self, schema: EntitySchema, fields: Sequence[str], *, unique: bool = False) -> bool:
        """Return whether this backend can honor one resolved scalar declaration.

        Args:
            schema: Entity schema owning the fields.
            fields: Ordered logical field names.
            unique: Whether the declaration is a correctness constraint.

        Returns:
            bool: Whether the type, arity, and requested semantics are supported.
        """
        caps = self.scalar_index()
        composite = len(fields) > 1
        if unique:
            if not caps.unique or (composite and not caps.composite_unique):
                return False
        elif not caps.value or (composite and not caps.composite):
            return False
        return all(
            type(schema.get(field_name).dtype).identifier in caps.supported_types
            and self.indexable(schema.get(field_name).dtype, resolver=self.module_resolver)
            for field_name in fields
        )

    def mcp(self) -> BackendMCP:
        """Return the complete backend MCP surface for this backend instance.

        Backend subclasses should override this method, call ``super().mcp()``, and add provider-specific
        tools/profiles to the returned ``BackendMCP`` object. This keeps callables, prompts/descriptions,
        profiles, default selection, and serializer configuration in one backend-owned contract.

        Args:
            None.

        Returns:
            BackendMCP: MCP specification containing generic backend inspection tools and profile defaults.

        Examples:
            Add a provider-specific read-only probe::

                def mcp(self) -> BackendMCP:
                    spec = super().mcp()
                    spec.tool("provider_probe", self.provider_probe, desc="Inspect provider-specific state.")
                    spec.profile("inspect", ("describe_backend", "backend_tags", "provider_probe"))
                    return spec
        """

        def describe_backend() -> dict[str, Any]:
            """Describe this backend instance.

            Args:
                None.

            Returns:
                dict[str, Any]: Backend metadata, effective Tag values, and
                registered backend MCP profiles.
            """
            return {
                "backend": self.to_dict(),
                "tags": backend_tags(),
                "profiles": spec.profile_summary(),
            }

        def backend_tags() -> dict[str, Any]:
            """Return effective Backend Tag metadata.

            Args:
                None.

            Returns:
                dict[str, Any]: Tag values with status, source, and snapshot evidence.
            """
            from ..utils.config.data import to_mutable_data

            return {
                identifier: {
                    "value": to_mutable_data(result.value),
                    "raw": to_mutable_data(result.raw),
                    "status": result.status,
                    "source": result.source,
                    "diagnostics": list(result.diagnostics),
                    "subject": list(result.subject),
                    "snapshot": result.snapshot,
                }
                for identifier, result in self.tags().items()
            }

        spec = BackendMCP(
            desc=f"HeavenBase backend {self.name!r} ({self.type}). Inspect physical backend metadata and provider-specific tools.",
            default_profile="inspect",
        )
        spec.tool(
            "describe_backend",
            describe_backend,
            desc="Describe the backend instance, provider type, effective Tags, and profiles.",
        )
        spec.tool("backend_tags", backend_tags, desc="Return effective Backend Tags with provenance and diagnostics.")
        spec.profile("inspect", ("describe_backend", "backend_tags"), desc="Read-only backend metadata inspection.")
        spec.profile("full", tuple(spec.tools), desc="All backend MCP tools.")
        return spec

    def to_mcp(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        profile: Any = None,
        tools: Iterable[str] | None = None,
    ) -> Any:
        """Return a native Toolkit bound to this backend instance.

        Args:
            name: Optional Toolkit name. Defaults to ``backend-<backend.name>`` or the configured
                ``heavenbase.mcp.backend_toolkit`` value.
            description: Optional Toolkit description override. Defaults to the description from
                ``self.mcp()`` plus the selected tool list.
            profile: Optional backend MCP profile selection. Accepts ``None`` for the backend default,
                a registered profile name, ``BackendMCPProfile``, or a mapping with ``name`` / ``tools`` /
                ``desc`` / ``description`` / ``prompt`` keys.
            tools: Optional explicit tool allowlist. When supplied, this overrides ``profile`` and creates
                an ad-hoc custom profile.

        Returns:
            Any: Runtime-only ``Toolkit`` bound to this backend instance.

        Examples:
            Expose only Tag inspection::

                toolkit = backend.to_mcp(tools=["backend_tags"])
        """
        from heavenbase.extensions.contributions import resolve_contribution

        backend_toolkit = resolve_contribution(
            "toolkit_function",
            "backend_toolkit",
            resolver=self.module_resolver,
            callable_only=True,
        )

        return backend_toolkit(
            self,
            name=name,
            description=description,
            profile=profile,
            tools=tools,
            resolver=self.module_resolver,
        )

    def to_mcp_json(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        profile: Any = None,
        tools: Iterable[str] | None = None,
        transport: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> str:
        """Return MCP client config JSON for this backend instance.

        Args:
            name: Optional Toolkit/server name.
            description: Optional Toolkit description override.
            profile: Optional backend MCP profile selection accepted by ``to_mcp``.
            tools: Optional explicit tool allowlist.
            transport: MCP transport. Defaults through ``heavenbase.mcp.transport``.
            host: HTTP host included in generated MCP client config.
            port: HTTP port included in generated MCP client config.

        Returns:
            str: Serialized MCP client configuration JSON.
        """
        from heavenbase.extensions.contributions import resolve_contribution

        mcp_transport = resolve_contribution("toolkit_function", "mcp_transport", resolver=self.module_resolver, callable_only=True)
        mcp_host = resolve_contribution("toolkit_function", "mcp_host", resolver=self.module_resolver, callable_only=True)
        mcp_port = resolve_contribution("toolkit_function", "mcp_port", resolver=self.module_resolver, callable_only=True)

        return self.to_mcp(name=name, description=description, profile=profile, tools=tools).to_mcp_json(
            transport=mcp_transport(transport, config=self.config),
            host=mcp_host(host, config=self.config),
            port=mcp_port(port, config=self.config),
        )

    def serve(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        profile: Any = None,
        tools: Iterable[str] | None = None,
        transport: str | None = None,
        host: str | None = None,
        port: int | None = None,
        **server_kwargs: Any,
    ) -> Any:
        """Serve this backend instance as an MCP server.

        Args:
            name: Optional Toolkit/server name.
            description: Optional Toolkit description override.
            profile: Optional backend MCP profile selection accepted by ``to_mcp``.
            tools: Optional explicit tool allowlist.
            transport: MCP transport. Defaults through ``heavenbase.mcp.transport``.
            host: HTTP host. Defaults through ``heavenbase.mcp.host``.
            port: HTTP port. Defaults through ``heavenbase.mcp.port``.
            **server_kwargs: Additional keyword arguments forwarded to FastMCP ``run``.

        Returns:
            Any: FastMCP result after the blocking server lifecycle completes.
        """
        from heavenbase.extensions.contributions import resolve_contribution

        mcp_transport = resolve_contribution("toolkit_function", "mcp_transport", resolver=self.module_resolver, callable_only=True)
        mcp_host = resolve_contribution("toolkit_function", "mcp_host", resolver=self.module_resolver, callable_only=True)
        mcp_port = resolve_contribution("toolkit_function", "mcp_port", resolver=self.module_resolver, callable_only=True)

        return self.to_mcp(name=name, description=description, profile=profile, tools=tools).serve(
            transport=mcp_transport(transport, config=self.config),
            host=mcp_host(host, config=self.config),
            port=mcp_port(port, config=self.config),
            **server_kwargs,
        )
